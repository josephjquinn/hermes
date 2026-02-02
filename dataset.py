
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from mask import available_tiles, damage_ratio, load_image, load_mask


def discover_disasters(data_root: Path) -> list[str]:
    data_root = Path(data_root)
    if not data_root.exists():
        return []
    disasters = []
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and (d / "masks").exists() and (d / "images").exists():
            disasters.append(d.name)
    return disasters


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DamageRatioDataset(Dataset):
    """Dataset of post-disaster images with damage ratio labels (0-1)."""

    IMAGENET_MEAN = IMAGENET_MEAN
    IMAGENET_STD = IMAGENET_STD

    def __init__(
        self,
        data_root: Path | str,
        disasters: list[str] | None = None,
        size: int = 224,
        transform: str = "train",
    ):
        self.data_root = Path(data_root)
        self.size = size
        self.transform = transform
        self._transforms = get_train_transforms(size) if transform == "train" else get_eval_transforms(size)

        if disasters is None:
            disasters = discover_disasters(self.data_root)
        self.disasters = disasters

        self.samples: list[tuple[str, str, float]] = []
        for disaster in self.disasters:
            tiles = available_tiles(self.data_root, disaster)
            base = self.data_root / disaster
            for tile_id in tiles:
                pre_mask_path = base / "masks" / f"{disaster}_{tile_id}_pre_disaster.png"
                post_mask_path = base / "masks" / f"{disaster}_{tile_id}_post_disaster.png"
                post_img_path = base / "images" / f"{disaster}_{tile_id}_post_disaster.png"
                if not post_img_path.exists() or not pre_mask_path.exists() or not post_mask_path.exists():
                    continue
                pre_mask = load_mask(pre_mask_path)
                post_mask = load_mask(post_mask_path)
                if pre_mask is None or post_mask is None:
                    continue
                ratio = damage_ratio(pre_mask, post_mask)
                self.samples.append((disaster, tile_id, ratio))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        disaster, tile_id, ratio = self.samples[idx]
        base = self.data_root / disaster
        img_path = base / "images" / f"{disaster}_{tile_id}_post_disaster.png"
        img = load_image(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self._transforms(img)
        return img, float(ratio)
