from pathlib import Path

import numpy as np
import torch

from model.model import DamageSegmentationModel

CLASS_NAMES = ["background", "no_damage", "minor", "major", "destroyed"]


def get_device(override: str | None = None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[DamageSegmentationModel, dict]: 
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
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    config = {"encoder": encoder, "size": size, "num_classes": num_classes}
    return model, config

def mean_iou_np(pred: np.ndarray, target: np.ndarray, num_classes: int) -> tuple[float, list[float]]:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    ious = []
    for c in range(num_classes):
        inter = np.logical_and(pred == c, target == c).sum()
        union = np.logical_or(pred == c, target == c).sum()
        ious.append(float(inter) / float(union) if union > 0 else 1.0)
    return (sum(ious) / num_classes) if ious else 0.0, ious


def pixel_accuracy_np(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred.ravel() == target.ravel()).mean())

def mean_iou_torch(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = -100) -> float:
    pred = pred.view(-1)
    target = target.view(-1)
    if ignore_index >= 0:
        valid = target != ignore_index
        pred = pred[valid]
        target = target[valid]
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        inter = (pred_c & target_c).long().sum().item()
        union = (pred_c | target_c).long().sum().item()
        ious.append(inter / union if union > 0 else 1.0)
    return sum(ious) / num_classes if ious else 0.0


def pixel_accuracy_torch(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred.view(-1) == target.view(-1)).float().mean().item()
