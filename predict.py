import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image

from dataset import get_eval_transforms
from model.model import DamageClassifier


def preprocess(image_path: Path, size: int = 224) -> torch.Tensor:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Failed to load: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    transform = get_eval_transforms(size)
    x = transform(img).unsqueeze(0)
    return x


def main():
    parser = argparse.ArgumentParser(description="Predict damage ratio from post-disaster image")
    parser.add_argument("image", type=Path, help="Path to post-disaster image")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--model", type=str, default=None, choices=("b0", "b2"), help="Model variant (default: from checkpoint)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device or
        ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    variant = args.model or (ckpt.get("variant", "b0") if isinstance(ckpt, dict) else "b0")
    model = DamageClassifier(variant=variant, pretrained=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    x = preprocess(args.image).to(device)
    with torch.no_grad():
        ratio = model(x).item()

    print(f"Damage ratio: {ratio:.4f} ({100 * ratio:.2f}%)")


if __name__ == "__main__":
    main()
