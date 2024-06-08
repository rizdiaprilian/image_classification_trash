import torch
from torch import nn
from torchvision import models


def modelling(device: torch.device):
    # Setup the model with pretrained weights and send it to the target device
    weights = models.ResNet18_Weights.DEFAULT
    pretrained_resnet18 = models.resnet18(weights=weights)
    num_ftrs = pretrained_resnet18.fc.in_features
    pretrained_resnet18.fc = nn.Linear(num_ftrs, 6)

    return pretrained_resnet18
