import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DamageRatioDataset, discover_disasters
from model.model import DamageClassifier


def main():
    parser = argparse.ArgumentParser(description="Evaluate damage ratio classifier")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Data root")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    variant = ckpt.get("variant", "b0")
    model = DamageClassifier(variant=variant, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    disasters = discover_disasters(args.data)
    dataset = DamageRatioDataset(args.data, disasters=disasters, size=224, transform="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Samples: {len(dataset)}")

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            preds = model(images)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    mae = np.abs(preds - targets).mean()
    mse = ((preds - targets) ** 2).mean()
    rmse = np.sqrt(mse)
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    print("\n=== Metrics ===")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    print("\n=== Sample Predictions ===")
    indices = np.random.choice(len(dataset), min(10, len(dataset)), replace=False)
    for i in indices:
        disaster, tile_id, actual = dataset.samples[i]
        pred = preds[i]
        print(f"  {disaster}/{tile_id}: actual={actual:.3f}, pred={pred:.3f}, error={abs(pred-actual):.3f}")

    if not args.no_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(targets, preds, alpha=0.5, s=20)
        axes[0].plot([0, 1], [0, 1], "r--", label="Perfect")
        axes[0].set_xlabel("Actual Damage Ratio")
        axes[0].set_ylabel("Predicted Damage Ratio")
        axes[0].set_title(f"Predictions vs Actual (R²={r2:.3f})")
        axes[0].legend()
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)

        errors = preds - targets
        axes[1].hist(errors, bins=30, edgecolor="black", alpha=0.7)
        axes[1].axvline(0, color="r", linestyle="--")
        axes[1].set_xlabel("Prediction Error (pred - actual)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Error Distribution (MAE={mae:.3f})")

        plt.tight_layout()
        plt.savefig("eval_results.png", dpi=150)
        print(f"\nSaved plot to eval_results.png")
        plt.show()


if __name__ == "__main__":
    main()
