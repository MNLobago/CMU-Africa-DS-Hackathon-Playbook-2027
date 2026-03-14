import torch.nn as nn
from torchvision import models


class MaskClassifier(nn.Module):
    def __init__(self, dropout: float = 0.4):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1)   # single logit → sigmoid → probability
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)          # raw logit

    def unfreeze_backbone(self, layers_from_end: int = 3):
        """Unfreeze last N blocks of EfficientNet for fine-tuning."""
        blocks = list(self.backbone.features.children())
        for block in blocks[-layers_from_end:]:
            for param in block.parameters():
                param.requires_grad = True
        print(f"Unfroze last {layers_from_end} backbone blocks.")

class ResNetClassifier(nn.Module):
    def __init__(self, dropout: float = 0.4):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1)
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self, layers_from_end: int = 3):
        children = list(self.backbone.children())
        for child in children[-layers_from_end:]:
            for param in child.parameters():
                param.requires_grad = True
        print(f"Unfroze last {layers_from_end} ResNet layers.")

class ResNet101Classifier(nn.Module):
    def __init__(self, dropout: float = 0.4):
        super().__init__()
        backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1)
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_backbone(self, layers_from_end: int = 3):
        children = list(self.backbone.children())
        for child in children[-layers_from_end:]:
            for param in child.parameters():
                param.requires_grad = True
        print(f"Unfroze last {layers_from_end} ResNet101 layers.")