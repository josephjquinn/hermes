import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    efficientnet_b0,
    efficientnet_b2,
)

EFFICIENTNET_FEATURES = {
    "b0": 1280,
    "b2": 1408,
}


class DamageClassifier(nn.Module):
    def __init__(self, variant: str = "b2", pretrained: bool = True):
        super().__init__()
        self.variant = variant
        if variant == "b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
        elif variant == "b2":
            weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_b2(weights=weights)
        else:
            raise ValueError(f"Unknown variant: {variant}. Use 'b0' or 'b2'.")

        num_features = EFFICIENTNET_FEATURES[variant]
        dropout = 0.2 if variant == "b0" else 0.3
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(-1)
