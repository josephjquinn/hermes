import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    CLASS_NAMES,
    get_device,
    load_model,
    mean_iou_np,
    pixel_accuracy_np,
)
from dataset import DamageSegmentationDataset, discover_disasters
from mask import damage_colored_mask

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def main():
    parser = argparse.ArgumentParser(description="Evaluate damage segmentation model")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Data root")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--num-vis", type=int, default=4, help="Number of samples to visualize")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device)
    size = config["size"]
    num_classes = config["num_classes"]
    print(f"Loaded checkpoint: {args.checkpoint} (encoder={config['encoder']}, size={size})")

    disasters = discover_disasters(args.data)
    dataset = DamageSegmentationDataset(args.data, disasters=disasters, size=size, transform="eval")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Samples: {len(dataset)}")

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for pre_img, post_img, mask in tqdm(loader, desc="Evaluating"):
            pre_img = pre_img.to(device)
            post_img = post_img.to(device)
            logits = model(pre_img, post_img)
            pred = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(mask.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    miou, per_class_iou = mean_iou_np(preds, targets, num_classes)
    acc = pixel_accuracy_np(preds, targets)

    print("\n=== Metrics ===")
    print(f"mIoU: {miou:.4f}")
    print(f"Pixel accuracy: {acc:.4f}")
    print("Per-class IoU:")
    for i, name in enumerate(CLASS_NAMES[:num_classes]):
        print(f"  {name}: {per_class_iou[i]:.4f}")

    if not args.no_plot and len(dataset) > 0:
        num_vis = min(args.num_vis, len(dataset))
        indices = np.random.choice(len(dataset), num_vis, replace=False)
        fig, axes = plt.subplots(num_vis, 4, figsize=(16, 4 * num_vis))
        if num_vis == 1:
            axes = axes.reshape(1, -1)
        for row, idx in enumerate(indices):
            pre_t, post_t, mask_gt = dataset[idx]
            pre_np = np.clip(pre_t.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
            post_np = np.clip(post_t.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
            with torch.no_grad():
                logits = model(pre_t.unsqueeze(0).to(device), post_t.unsqueeze(0).to(device))
                pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            mask_gt_np = mask_gt.numpy()
            axes[row, 0].imshow(pre_np)
            axes[row, 0].set_title("Pre")
            axes[row, 0].axis("off")
            axes[row, 1].imshow(post_np)
            axes[row, 1].set_title("Post")
            axes[row, 1].axis("off")
            axes[row, 2].imshow(damage_colored_mask(pred_mask, bgr=False))
            axes[row, 2].set_title("Pred")
            axes[row, 2].axis("off")
            axes[row, 3].imshow(damage_colored_mask(mask_gt_np, bgr=False))
            axes[row, 3].set_title("GT")
            axes[row, 3].axis("off")
        plt.tight_layout()
        plt.savefig("eval_results.png", dpi=150)
        print(f"\nSaved plot to eval_results.png")
        plt.show()


if __name__ == "__main__":
    main()
