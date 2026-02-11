import argparse
import sys
from pathlib import Path

# Allow imports from repo root when run as script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model.common import get_device, mean_iou_torch, pixel_accuracy_torch
from model.dataset import DamageSegmentationDataset, discover_disasters
from model.model import DamageSegmentationModel


def compute_class_weights(train_loader: DataLoader, num_classes: int, device: torch.device, cap: float = 15.0) -> torch.Tensor | None:
    class_counts = [0] * num_classes
    for pre_img, post_img, mask in train_loader:
        for c in range(num_classes):
            class_counts[c] += (mask == c).sum().item()
    total = sum(class_counts)
    if total == 0:
        return None
    freq = [class_counts[c] / total for c in range(num_classes)]
    median_freq = np.median(freq)
    weights = []
    for f in freq:
        if f > 0:
            w = median_freq / f
            weights.append(min(w, cap))
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser(description="Train damage segmentation (pre+post -> mask)")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Data root")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"), help="Output dir for checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--test-frac", type=float, default=0.15, help="Test fraction (rest is train)")
    parser.add_argument("--encoder", type=str, default="resnet34", help="smp encoder name (e.g. resnet34, resnet50)")
    parser.add_argument("--size", type=int, default=256, help="Input spatial size (H, W)")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weights (use uniform CE)")
    parser.add_argument("--weight-cap", type=float, default=15.0, help="Cap class weight for stability")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    disasters = discover_disasters(args.data)
    if not disasters:
        raise SystemExit(f"No disasters found under {args.data}")

    dataset = DamageSegmentationDataset(args.data, disasters=disasters, size=args.size, transform="train")
    if len(dataset) == 0:
        raise SystemExit(f"No samples in dataset under {args.data}")

    n_val = int(len(dataset) * args.val_frac)
    n_test = int(len(dataset) * args.test_frac)
    n_train = len(dataset) - n_val - n_test
    if n_train < 0:
        n_train = len(dataset) - n_val
        n_test = 0
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
    print(f"Split: train={n_train}, val={n_val}, test={n_test}")

    device = get_device()
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if n_test > 0 else None

    if not args.no_class_weights:
        print("Computing class weights from training set...")
        weight_tensor = compute_class_weights(train_loader, args.num_classes, device, cap=args.weight_cap)
        if weight_tensor is not None:
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            print(f"Class weights: {[f'{w:.3f}' for w in weight_tensor.cpu().tolist()]}")
        else:
            criterion = nn.CrossEntropyLoss()
            print("Class weights: none (no pixels counted)")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Class weights: disabled")

    model = DamageSegmentationModel(
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=6,
        num_classes=args.num_classes,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output.mkdir(parents=True, exist_ok=True)
    best_val_miou = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_miou = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]")
        for pre_img, post_img, mask in pbar:
            pre_img = pre_img.to(device)
            post_img = post_img.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            logits = model(pre_img, post_img)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            miou = mean_iou_torch(pred, mask, args.num_classes)
            train_loss += loss.item() * pre_img.size(0)
            train_miou += miou * pre_img.size(0)
            n += pre_img.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", miou=f"{miou:.4f}")

        scheduler.step()
        train_loss /= n
        train_miou /= n

        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        val_acc = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for pre_img, post_img, mask in val_loader:
                pre_img = pre_img.to(device)
                post_img = post_img.to(device)
                mask = mask.to(device)
                logits = model(pre_img, post_img)
                loss = criterion(logits, mask)
                pred = logits.argmax(dim=1)
                miou = mean_iou_torch(pred, mask, args.num_classes)
                acc = pixel_accuracy_torch(pred, mask)
                val_loss += loss.item()
                val_miou += miou
                val_acc += acc
                n_val_batches += 1

        val_loss /= n_val_batches
        val_miou /= n_val_batches
        val_acc /= n_val_batches

        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} train_mIoU={train_miou:.4f} "
              f"val_loss={val_loss:.4f} val_mIoU={val_miou:.4f} val_acc={val_acc:.4f}")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            ckpt_path = args.output / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_miou": val_miou,
                "encoder": args.encoder,
                "size": args.size,
                "num_classes": args.num_classes,
            }, ckpt_path)
            print(f"  Saved best to {ckpt_path}")

    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "encoder": args.encoder,
        "size": args.size,
        "num_classes": args.num_classes,
    }, args.output / "last.pt")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(args.output / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Training done. Best val_mIoU={best_val_miou:.4f}")

    if test_loader is not None and (args.output / "best.pt").exists():
        print("\n=== Test set (best checkpoint) ===")
        ckpt = torch.load(args.output / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        test_loss = 0.0
        test_miou = 0.0
        test_acc = 0.0
        n_test_batches = 0
        with torch.no_grad():
            for pre_img, post_img, mask in test_loader:
                pre_img = pre_img.to(device)
                post_img = post_img.to(device)
                mask = mask.to(device)
                logits = model(pre_img, post_img)
                loss = criterion(logits, mask)
                pred = logits.argmax(dim=1)
                test_loss += loss.item()
                test_miou += mean_iou_torch(pred, mask, args.num_classes)
                test_acc += pixel_accuracy_torch(pred, mask)
                n_test_batches += 1
        test_loss /= n_test_batches
        test_miou /= n_test_batches
        test_acc /= n_test_batches
        print(f"Test loss:  {test_loss:.4f}")
        print(f"Test mIoU:  {test_miou:.4f}")
        print(f"Test acc:   {test_acc:.4f}")

if __name__ == "__main__":
    main()
