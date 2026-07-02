import torch
import torch.nn as nn
import torchvision.models.segmentation as segm
from torchvision.models.segmentation.deeplabv3 import ASPP

class LightASPPHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, atrous_rates=(12, 24)):
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class Model(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()

        # DeepLabv3 model with Mobilenetv3 backbone
        self.model = segm.deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,  
            aux_loss=False)
        
        in_channels = self.model.classifier[0].convs[0][0].in_channels
        
        # Replace last classifier for a ASPP-head   
        self.model.classifier = LightASPPHead(
            in_channels=in_channels,
            num_classes=n_classes,
            atrous_rates=(12, 24),  
        )

    def forward(self, x):
        out = self.model(x)
        return out["out"]