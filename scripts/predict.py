import argparse
import sys
from pathlib import Path

# Allow imports from repo root when run as script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.common import CLASS_NAMES, get_device, load_model, mean_iou_np, pixel_accuracy_np
from model.dataset import discover_disasters, get_segmentation_eval_transforms
from mask import damage_colored_mask, load_mask


def preprocess_pair(pre_path: Path, post_path: Path, size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    pre_img = cv2.imread(str(pre_path))
    post_img = cv2.imread(str(post_path))
    if pre_img is None:
        raise FileNotFoundError(f"Failed to load: {pre_path}")
    if post_img is None:
        raise FileNotFoundError(f"Failed to load: {post_path}")
    pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
    post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
    transform = get_segmentation_eval_transforms(size)
    out = transform(image=pre_img, post=post_img)
    pre_t = torch.from_numpy(out["image"].transpose(2, 0, 1)).float().unsqueeze(0)
    post_t = torch.from_numpy(out["post"].transpose(2, 0, 1)).float().unsqueeze(0)
    return pre_t, post_t


def run_one(model, pre_t: torch.Tensor, post_t: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        logits = model(pre_t.to(device), post_t.to(device))
        return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Predict damage mask from pre + post disaster images")
    parser.add_argument("pre_image", type=Path, nargs="?", default=None, help="Path to pre-disaster image")
    parser.add_argument("post_image", type=Path, nargs="?", default=None, help="Path to post-disaster image")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--output", type=Path, default=None, help="Output path for mask PNG (single pair mode)")
    parser.add_argument("--colorized", action="store_true", help="Save colorized mask (default: raw single-channel)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--random", type=int, default=None, metavar="N", help="Pick N random samples from --data")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Data root (for --random)")
    parser.add_argument("--output-dir", type=Path, default=Path("predict_output"), help="Output dir when using --random --save")
    parser.add_argument("--save", action="store_true", help="Save masks to --output-dir when using --random")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for --random")
    args = parser.parse_args()

    device = get_device(args.device)
    model, config = load_model(args.checkpoint, device)
    size = config["size"]
    num_classes = config["num_classes"]

    if args.random is not None:
        _run_random_samples(args, model, device, size, num_classes)
        return

    if args.pre_image is None or args.post_image is None:
        raise SystemExit("Provide pre_image and post_image paths, or use --random N with --data")
    pre_t, post_t = preprocess_pair(args.pre_image, args.post_image, size=size)
    pred = run_one(model, pre_t, post_t, device)

    print(f"Damage mask shape: {pred.shape}, unique classes: {np.unique(pred).tolist()}")
    out_path = args.output or Path("predict_mask.png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.colorized or args.output is None:
        cv2.imwrite(str(out_path), cv2.cvtColor(damage_colored_mask(pred, bgr=True), cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(out_path), pred)
    print(f"Saved mask to {out_path}")

    for c, name in enumerate(CLASS_NAMES[:num_classes]):
        count = (pred == c).sum()
        print(f"  {name}: {count} px ({100.0 * count / pred.size:.1f}%)")


def _run_random_samples(args, model, device, size: int, num_classes: int):
    if args.random < 1:
        raise SystemExit("--random must be >= 1")
    if args.seed is not None:
        np.random.seed(args.seed)

    from model.dataset import DamageSegmentationDataset

    disasters = discover_disasters(args.data)
    if not disasters:
        raise SystemExit(f"No disasters found under {args.data}")
    dataset = DamageSegmentationDataset(args.data, disasters=disasters, size=size, transform="eval")
    if len(dataset) == 0:
        raise SystemExit("No samples in dataset")

    n = min(args.random, len(dataset))
    indices = np.random.choice(len(dataset), n, replace=False)
    if args.save:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_images = []
    truth_images = []
    titles = []
    print(f"Running on {n} random samples...")
    for i, idx in enumerate(indices):
        disaster, tile_id = dataset.samples[idx]
        base = args.data / disaster
        pre_path = base / "images" / f"{disaster}_{tile_id}_pre_disaster.png"
        post_path = base / "images" / f"{disaster}_{tile_id}_post_disaster.png"
        post_mask_path = base / "masks" / f"{disaster}_{tile_id}_post_disaster.png"

        pre_t, post_t = preprocess_pair(pre_path, post_path, size=size)
        pred = run_one(model, pre_t, post_t, device)
        pred_rgb = damage_colored_mask(pred, bgr=False)
        pred_images.append(pred_rgb)

        truth = load_mask(post_mask_path)
        if truth is not None:
            truth = np.clip(truth.astype(np.int64), 0, 4)
            truth = cv2.resize(truth.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
            truth_rgb = damage_colored_mask(truth, bgr=False)
        else:
            truth = np.zeros((size, size), dtype=np.uint8)
            truth_rgb = np.zeros((size, size, 3), dtype=np.uint8)
        truth_images.append(truth_rgb)

        acc = pixel_accuracy_np(pred, truth)
        miou, _ = mean_iou_np(pred, truth, num_classes)
        titles.append(f"{disaster}\n{tile_id}\nacc: {acc:.3f}  mIoU: {miou:.3f}")
        print(f"  [{i+1}/{n}] {disaster}/{tile_id}  acc={acc:.3f}  mIoU={miou:.3f}")

        if args.save:
            out_path = args.output_dir / f"{disaster}_{tile_id}_mask.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
    if args.save:
        print(f"Saved {n} masks to {args.output_dir}")
    
    samples_per_row = min(n, 5)
    ncols = 2 * samples_per_row
    nrows = (n + samples_per_row - 1) // samples_per_row
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        r, pair = i // samples_per_row, i % samples_per_row
        axes[r, 2 * pair].imshow(pred_images[i])
        axes[r, 2 * pair].set_title("Pred" if i == 0 else "", fontsize=10)
        axes[r, 2 * pair].set_ylabel(titles[i], fontsize=8)
        axes[r, 2 * pair].axis("off")
        axes[r, 2 * pair + 1].imshow(truth_images[i])
        axes[r, 2 * pair + 1].set_title("Truth" if i == 0 else "", fontsize=10)
        axes[r, 2 * pair + 1].axis("off")
    for i in range(n, nrows * samples_per_row):
        r, pair = i // samples_per_row, i % samples_per_row
        for j in [0, 1]:
            axes[r, 2 * pair + j].axis("off")
    plt.suptitle("Predicted vs ground truth damage masks", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
