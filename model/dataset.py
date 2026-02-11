from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from mask import available_tiles, damage_ratio, load_image, load_mask

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False


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


def get_segmentation_train_transforms(size: int = 256):
    """Albumentations pipeline: same geometric transform for pre image, post image, and mask."""
    if not _HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for DamageSegmentationDataset")
    return A.Compose(
        [
            A.Resize(height=size, width=size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
        additional_targets={"post": "image"},
    )


def get_segmentation_eval_transforms(size: int = 256):
    """No augmentation; resize and normalize for validation/inference."""
    if not _HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required for DamageSegmentationDataset")
    return A.Compose(
        [
            A.Resize(height=size, width=size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
        additional_targets={"post": "image"},
    )


class DamageSegmentationDataset(Dataset):
    NUM_CLASSES = 5 

    def __init__(
        self,
        data_root: Path | str,
        disasters: list[str] | None = None,
        size: int = 256,
        transform: str = "train",
    ):
        self.data_root = Path(data_root)
        self.size = size
        self.transform_mode = transform
        self._transform = (
            get_segmentation_train_transforms(size) if transform == "train" else get_segmentation_eval_transforms(size)
        )

        if disasters is None:
            disasters = discover_disasters(self.data_root)
        self.disasters = disasters

        self.samples: list[tuple[str, str]] = []  # (disaster, tile_id)
        for disaster in self.disasters:
            tiles = available_tiles(self.data_root, disaster)
            base = self.data_root / disaster
            for tile_id in tiles:
                pre_img_path = base / "images" / f"{disaster}_{tile_id}_pre_disaster.png"
                post_img_path = base / "images" / f"{disaster}_{tile_id}_post_disaster.png"
                pre_mask_path = base / "masks" / f"{disaster}_{tile_id}_pre_disaster.png"
                post_mask_path = base / "masks" / f"{disaster}_{tile_id}_post_disaster.png"
                if not pre_img_path.exists() or not post_img_path.exists():
                    continue
                if not pre_mask_path.exists() or not post_mask_path.exists():
                    continue
                post_mask = load_mask(post_mask_path)
                if post_mask is None:
                    continue
                self.samples.append((disaster, tile_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        disaster, tile_id = self.samples[idx]
        base = self.data_root / disaster
        pre_img = load_image(base / "images" / f"{disaster}_{tile_id}_pre_disaster.png")
        post_img = load_image(base / "images" / f"{disaster}_{tile_id}_post_disaster.png")
        post_mask = load_mask(base / "masks" / f"{disaster}_{tile_id}_post_disaster.png")
        if pre_img is None or post_img is None or post_mask is None:
            raise FileNotFoundError(f"Missing data for {disaster}/{tile_id}")

        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
        post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
        post_mask = np.clip(post_mask.astype(np.int64), 0, 4)

        transformed = self._transform(image=pre_img, post=post_img, mask=post_mask)
        pre_t = torch.from_numpy(transformed["image"].transpose(2, 0, 1)).float()
        post_t = torch.from_numpy(transformed["post"].transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(transformed["mask"]).long().clamp(0, 4)
        return pre_t, post_t, mask_t


class DamageRatioDataset(Dataset):
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
