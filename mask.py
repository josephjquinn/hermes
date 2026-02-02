import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import argparse
import random


BACKGROUND = 0
NO_DAMAGE = 1
MINOR_DAMAGE = 2
MAJOR_DAMAGE = 3
DESTROYED = 4

# pre distaster
# 0: background, 1: building
BUILDING = 1

# post distaster
DAMAGE_COLORS_BGR = {
    BACKGROUND: [0, 0, 0],       # black
    NO_DAMAGE: [0, 255, 0],     # green
    MINOR_DAMAGE: [255, 255, 0], # yellow
    MAJOR_DAMAGE: [0, 165, 255], # orange
    DESTROYED: [0, 0, 255],     # red
}


def load_image(path: str | Path) -> np.ndarray:
    return cv2.imread(str(path))


def load_mask(path: str | Path, normalize_255: bool = False) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if normalize_255 and mask is not None:
        mask = np.where(mask == 255, 1, mask)
    return mask


def load_pair(
    base_path: str | Path,
    disaster: str,
    tile_id: str,
    *,
    images_dir: str = "images",
    masks_dir: str = "masks",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = Path(base_path) / disaster
    pre_image = load_image(base / images_dir / f"{disaster}_{tile_id}_pre_disaster.png")
    post_image = load_image(base / images_dir / f"{disaster}_{tile_id}_post_disaster.png")
    pre_mask = load_mask(base / masks_dir / f"{disaster}_{tile_id}_pre_disaster.png")
    post_mask = load_mask(base / masks_dir / f"{disaster}_{tile_id}_post_disaster.png")
    return pre_image, post_image, pre_mask, post_mask


def damage_ratio(pre_mask: np.ndarray, post_mask: np.ndarray) -> float:
    building_pixels = (pre_mask == BUILDING) | (pre_mask == 255)
    if building_pixels.sum() == 0:
        return 0.0
    damaged = (post_mask == MAJOR_DAMAGE) | (post_mask == DESTROYED)
    return float((building_pixels & damaged).sum()) / float(building_pixels.sum())


def damage_counts(mask: np.ndarray) -> dict[str, int]:
    return {
        "no_damage": int(np.sum(mask == NO_DAMAGE)),
        "minor_damage": int(np.sum(mask == MINOR_DAMAGE)),
        "major_damage": int(np.sum(mask == MAJOR_DAMAGE)),
        "destroyed": int(np.sum(mask == DESTROYED)),
    }


def damage_colored_mask(mask: np.ndarray, bgr: bool = True) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for value, color in DAMAGE_COLORS_BGR.items():
        out[mask == value] = color
    if not bgr:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


def unique_levels(mask: np.ndarray) -> np.ndarray:
    return np.unique(mask)


def visualize(
    pre_image: np.ndarray,
    pre_mask: np.ndarray,
    post_image: np.ndarray,
    post_mask: np.ndarray,
    *,
    figsize: tuple[float, float] = (12, 12),
):

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes[0, 0].imshow(cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Pre-Disaster Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Post-Disaster Image")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(pre_mask, cmap="viridis")
    axes[1, 0].set_title("Pre-Disaster Mask (Buildings)")
    axes[1, 0].axis("off")

    colored = damage_colored_mask(post_mask, bgr=False)
    axes[1, 1].imshow(colored)
    axes[1, 1].set_title("Post-Disaster Damage")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


def available_tiles(data_root: Path, disaster: str, masks_dir: str = "masks") -> list[str]:
    mask_dir = data_root / disaster / masks_dir
    if not mask_dir.exists():
        return []
    prefix = f"{disaster}_"
    suffix = "_post_disaster.png"
    tiles = []
    for p in mask_dir.glob(f"{prefix}*{suffix}"):
        name = p.name
        if name.startswith(prefix) and name.endswith(suffix):
            tile_id = name[len(prefix) : -len(suffix)]
            tiles.append(tile_id)
    return sorted(tiles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask PNG format: load, stats, visualize")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Data root")
    parser.add_argument("--disaster", type=str, default="EARTHQUAKE-TURKEY")
    parser.add_argument("--tile", type=str, default=None, help="Tile ID (default: random from disaster)")
    parser.add_argument("--no-show", action="store_true", help="Print stats only, no plot")
    args = parser.parse_args()

    base = args.data / args.disaster
    tiles = available_tiles(args.data, args.disaster)
    if not tiles:
        print(f"No tiles found for {args.disaster} under {args.data}")
        raise SystemExit(1)
    tile = args.tile if args.tile is not None else random.choice(tiles)
    if args.tile is None:
        print(f"Random tile: {tile}")

    pre_img_path = base / "images" / f"{args.disaster}_{tile}_pre_disaster.png"
    post_img_path = base / "images" / f"{args.disaster}_{tile}_post_disaster.png"
    pre_mask_path = base / "masks" / f"{args.disaster}_{tile}_pre_disaster.png"
    post_mask_path = base / "masks" / f"{args.disaster}_{tile}_post_disaster.png"

    if not post_mask_path.exists():
        print(f"Not found: {post_mask_path}")
        raise SystemExit(1)

    pre_mask = load_mask(pre_mask_path)
    post_mask = load_mask(post_mask_path)
    print(f"Pre-disaster buildings: {unique_levels(pre_mask).tolist()}")
    print(f"Post-disaster damage levels: {unique_levels(post_mask).tolist()}")
    print("Damage counts:", damage_counts(post_mask))

    if not args.no_show and pre_img_path.exists() and post_img_path.exists():
        pre_image = load_image(pre_img_path)
        post_image = load_image(post_img_path)
        visualize(pre_image, pre_mask, post_image, post_mask)
