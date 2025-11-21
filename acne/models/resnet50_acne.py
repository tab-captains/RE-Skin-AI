# acne/models/resnet50_acne.py
import torch.nn as nn
from torchvision import models


def build_resnet50_acne(num_classes: int, freeze_backbone: bool = False):
    """
    ImageNet pretrained ResNet50을 여드름 분류용으로 커스터마이즈.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
