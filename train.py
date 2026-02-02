import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import DamageRatioDataset, discover_disasters
from model.model import DamageClassifier


def main():
    parser = argparse.ArgumentParser(description="Train damage ratio classifier")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Data root")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"), help="Output dir for checkpoints")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--model", type=str, default="b2", choices=("b0", "b2"), help="EfficientNet variant")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    disasters = discover_disasters(args.data)
    if not disasters:
        raise SystemExit(f"No disasters found under {args.data}")

    dataset = DamageRatioDataset(args.data, disasters=disasters, size=224, transform="train")
    if len(dataset) == 0:
        raise SystemExit(f"No samples in dataset under {args.data}")

    n_val = int(len(dataset) * args.val_frac)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DamageClassifier(variant=args.model, pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.float().to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_mae += (preds - targets).abs().sum().item()
            n += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss /= n
        train_mae /= n

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.float().to(device)
                preds = model(images)
                loss = criterion(preds, targets)
                val_loss += loss.item()
                val_mae += (preds - targets).abs().mean().item()
                n_val_batches += 1

        val_loss /= n_val_batches
        val_mae /= n_val_batches

        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} train_mae={train_mae:.4f} "
              f"val_loss={val_loss:.4f} val_mae={val_mae:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = args.output / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "variant": args.model,
            }, ckpt_path)
            print(f"  Saved best to {ckpt_path}")

    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "variant": args.model,
    }, args.output / "last.pt")
    with open(args.output / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Training done. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
