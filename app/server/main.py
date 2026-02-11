import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .routing import route_order, total_distance_km

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.dataset import get_segmentation_eval_transforms
from mask import damage_colored_mask, damage_counts
from model.model import DamageSegmentationModel

app = FastAPI(
    title="Orthanc Damage Segmentation API",
    description="Submit pre- and post-disaster image pairs to get damage segmentation masks.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_device = None
_size = 256
_num_classes = 5


def _load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder = ckpt.get("encoder", "resnet34")
    size = ckpt.get("size", 256)
    num_classes = ckpt.get("num_classes", 5)
    model = DamageSegmentationModel(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=6,
        num_classes=num_classes,
    )
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model, size, num_classes


@app.on_event("startup")
def startup():
    global _model, _device, _size, _num_classes
    checkpoint = ROOT / "checkpoints" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. "
            "Train the model first or set CHECKPOINT_PATH to a valid path."
        )
    _device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    _model, _size, _num_classes = _load_model(checkpoint, _device)


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    return img


def _preprocess_pair(pre_img: np.ndarray, post_img: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    pre_rgb = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
    post_rgb = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
    transform = get_segmentation_eval_transforms(_size)
    transformed = transform(image=pre_rgb, post=post_rgb)
    pre_t = torch.from_numpy(transformed["image"].transpose(2, 0, 1)).float().unsqueeze(0)
    post_t = torch.from_numpy(transformed["post"].transpose(2, 0, 1)).float().unsqueeze(0)
    return pre_t, post_t


def _run_inference(pre_t: torch.Tensor, post_t: torch.Tensor) -> np.ndarray:
    pre_t = pre_t.to(_device)
    post_t = post_t.to(_device)
    with torch.no_grad():
        logits = _model(pre_t, post_t)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def _calculate_damage_score(pred: np.ndarray, counts: dict) -> float:
    damage_weights = {
        "no_damage": 0.0,
        "minor_damage": 25.0,
        "major_damage": 75.0,
        "destroyed": 100.0,
    }
    total_building_pixels = sum(counts.values())
    if total_building_pixels == 0:
        return 0.0
    weighted_sum = 0.0
    for category, weight in damage_weights.items():
        pixel_count = counts.get(category, 0)
        weighted_sum += pixel_count * weight
    damage_score = weighted_sum / total_building_pixels

    return round(damage_score, 2)


class RouteHub(BaseModel):
    lat: float
    lng: float


class RouteSite(BaseModel):
    lat: float
    lng: float
    damage_score: float


class RouteRequest(BaseModel):
    hub: RouteHub
    sites: list[RouteSite]
    damage_weight: float = 1.0
    algorithm: str = "greedy"


class RouteResponse(BaseModel):
    order: list[int]
    total_distance_km: float
    total_cost_km: float
    damage_weight: float = 1.0
    algorithm: str = "greedy"


class SummarySite(BaseModel):
    label: str
    lat: float
    lng: float
    damage_score: float
    stats: dict[str, dict[str, float | int]]


class SummaryRequest(BaseModel):
    hub: RouteHub | None
    sites: list[SummarySite]
    route_order: list[int] | None
    total_distance_km: float | None
    total_cost_km: float | None


@app.post("/route", response_model=RouteResponse)
def compute_route(req: RouteRequest) -> RouteResponse:
    if not req.sites:
        return RouteResponse(
            order=[0], total_distance_km=0.0, total_cost_km=0.0,
            damage_weight=req.damage_weight, algorithm=req.algorithm,
        )
    algo = req.algorithm if req.algorithm in ("greedy", "tsp") else "greedy"
    hub = (req.hub.lat, req.hub.lng)
    sites = [(s.lat, s.lng, s.damage_score) for s in req.sites]
    order, cost_km = route_order(hub, sites, damage_weight=req.damage_weight, algorithm=algo)
    coords = [hub] + [(s.lat, s.lng) for s in req.sites]
    dist_km = total_distance_km(order, coords)
    return RouteResponse(
        order=order,
        total_distance_km=round(dist_km, 4),
        total_cost_km=round(cost_km, 4),
        damage_weight=req.damage_weight,
        algorithm=algo,
    )


@app.post("/summary")
def generate_summary(req: SummaryRequest) -> dict[str, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not set. Set it in your environment to enable summaries.",
        )
    client = OpenAI(api_key=api_key)

    def _fmt_stats(stats: dict) -> str:
        parts = []
        for k, v in (stats or {}).items():
            if k == "background":
                continue
            pct = v.get("percent", 0) if isinstance(v, dict) else v
            parts.append(f"{k}={pct}%")
        return ", ".join(parts)

    sites_text = "\n".join(
        f"- {s.label}: damage score {s.damage_score}/100, lat {s.lat:.4f} lng {s.lng:.4f}. Stats: {_fmt_stats(s.stats)}"
        for s in req.sites
    )
    routing_text = ""
    if req.route_order and len(req.route_order) > 1 and req.hub:
        visit_order = [
            f"Stop {i}: {req.sites[idx - 1].label} (score {req.sites[idx - 1].damage_score})"
            for i, idx in enumerate(req.route_order[1:], 1)
        ]
        routing_text = (
            f"Routing starts at hub ({req.hub.lat:.4f}, {req.hub.lng:.4f}). "
            f"Visit order: {' → '.join(visit_order)}. "
        )
        if req.total_distance_km is not None:
            routing_text += f"Total distance: {req.total_distance_km:.2f} km. "
        if req.total_cost_km is not None:
            routing_text += f"Damage-weighted cost: {req.total_cost_km:.2f} km."
    else:
        routing_text = "No routing was computed (single site or no hub provided)."

    prompt = f"""You are summarizing disaster damage assessment results for emergency responders.

DAMAGE MASKS:
Each location was analyzed with a segmentation model that classifies building pixels into:
- background (not buildings)
- no_damage (0-25%): intact structures
- minor_damage (25%): light damage
- major_damage (75%): severe damage  
- destroyed (100%): complete destruction

The damage score (0-100) is a weighted average across building pixels.

SITES ASSESSED:
{sites_text}

ROUTING:
{routing_text}

The routing uses A* search with damage-weighted cost: visiting higher-damage sites sooner reduces the effective travel cost. Distance is multiplied by (1 - damage_score/100), so severely damaged sites are prioritized.

Write a concise 2-4 paragraph summary that:
1. Explains what the damage masks show at each location (severity, key stats)
2. Explains the routing order and why sites were prioritized that way
3. Gives actionable guidance for responders

Use plain language. Be specific about scores and percentages."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        summary = resp.choices[0].message.content or ""
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {str(e)}")


_N_SEED_STOPS = 7

_COLORADO_HUB_LAT = 39.7392
_COLORADO_HUB_LNG = -104.9903
_COLORADO_IMAGE_IDS = [
    "000013", "000248", "008382", "009307", "000042", "000092", "000147",
]

_JAPAN_HUB_LAT = 35.6762
_JAPAN_HUB_LNG = 139.6503
_JAPAN_IMAGE_IDS = [
    "000092", "009307", "000147", "008382", "000013", "000042", "000248",
    "000059", "000102", "000120", "000161", "000191", "000236", "000295",
]
_JAPAN_SEED_OFFSETS: list[tuple[float, float]] = [
    (0.072, 0.018),
    (0.031, 0.089),
    (-0.058, 0.041),
    (-0.024, -0.076),
    (0.085, -0.035),
    (-0.047, 0.062),
    (0.012, -0.091),
    (0.098, 0.055),
    (-0.066, -0.052),
    (0.041, -0.088),
    (-0.079, 0.027),
    (0.053, 0.071),
    (-0.038, -0.094),
    (0.089, -0.061),
]

_SEED_OFFSETS: list[tuple[float, float]] = [
    (0.072, 0.018),
    (0.031, 0.089),
    (-0.058, 0.041),
    (-0.024, -0.076),
    (0.085, -0.035),
    (-0.047, 0.062),
    (0.012, -0.091),
]


def _seed_coords_from_hub(
    hub_lat: float, hub_lng: float, offsets: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    return [(hub_lat + dlat, hub_lng + dlng) for dlat, dlng in offsets]


def _build_seed_response(
    data_dir: Path,
    coords: list[tuple[float, float]],
    hub_lat: float,
    hub_lng: float,
    image_ids: list[str],
) -> dict:
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail="Seed data directory not found")
    entries = []
    for i in range(len(coords)):
        img_id = image_ids[i % len(image_ids)]
        pre_path = data_dir / f"EARTHQUAKE-TURKEY_{img_id}_pre_disaster.png"
        post_path = data_dir / f"EARTHQUAKE-TURKEY_{img_id}_post_disaster.png"
        if not pre_path.exists() or not post_path.exists():
            raise HTTPException(status_code=404, detail=f"Seed image pair not found: {img_id}")
        lat, lng = coords[i]
        pre_b64 = base64.b64encode(pre_path.read_bytes()).decode("ascii")
        post_b64 = base64.b64encode(post_path.read_bytes()).decode("ascii")
        entries.append({
            "pre_base64": pre_b64,
            "post_base64": post_b64,
            "lat": lat,
            "lng": lng,
        })
    return {
        "entries": entries,
        "hub": {"lat": hub_lat, "lng": hub_lng},
    }


@app.get("/seed/colorado")
def seed_colorado():
    data_dir = ROOT / "data" / "EARTHQUAKE-TURKEY" / "images"
    coords = _seed_coords_from_hub(
        _COLORADO_HUB_LAT, _COLORADO_HUB_LNG, _SEED_OFFSETS
    )
    return _build_seed_response(
        data_dir, coords, _COLORADO_HUB_LAT, _COLORADO_HUB_LNG, _COLORADO_IMAGE_IDS
    )


@app.get("/seed/japan")
def seed_japan():
    data_dir = ROOT / "data" / "EARTHQUAKE-TURKEY" / "images"
    coords = _seed_coords_from_hub(
        _JAPAN_HUB_LAT, _JAPAN_HUB_LNG, _JAPAN_SEED_OFFSETS
    )
    return _build_seed_response(
        data_dir, coords, _JAPAN_HUB_LAT, _JAPAN_HUB_LNG, _JAPAN_IMAGE_IDS
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
async def predict(
    pre_image: UploadFile = File(..., description="Pre-disaster image"),
    post_image: UploadFile = File(..., description="Post-disaster image"),
):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    pre_data = await pre_image.read()
    post_data = await post_image.read()
    pre_img = _decode_image(pre_data)
    post_img = _decode_image(post_data)
    pre_t, post_t = _preprocess_pair(pre_img, post_img)
    pred = _run_inference(pre_t, post_t)
    colorized = damage_colored_mask(pred, bgr=True)
    _, png_bytes = cv2.imencode(".png", colorized)
    mask_b64 = base64.b64encode(png_bytes.tobytes()).decode("ascii")
    counts = damage_counts(pred)
    total = pred.size
    class_names = ["background", "no_damage", "minor_damage", "major_damage", "destroyed"]
    background_px = int((pred == 0).sum())
    stats = {}
    stats["background"] = {"pixels": background_px, "percent": round(100.0 * background_px / total, 2)}
    for name in class_names[1:_num_classes]:
        px = counts.get(name, 0)
        stats[name] = {"pixels": px, "percent": round(100.0 * px / total, 2)}
    damage_score = _calculate_damage_score(pred, counts)

    return {
        "mask_image_base64": mask_b64,
        "shape": list(pred.shape),
        "damage_counts": counts,
        "stats": stats,
        "damage_score": damage_score,
    }
