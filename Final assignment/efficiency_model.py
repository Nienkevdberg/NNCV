import torch
import torch.nn as nn
import torchvision.models.segmentation as segm


class Model(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()

        # DeepLabv3 model with Mobilenetv3 backbone
        self.model = segm.deeplabv3_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,  
            aux_loss=False)
        
        # Replace last classifier to match number of classes    
        self.model.classifier[-1] = nn.Conv2d(
            in_channels=256,
            out_channels=n_classes,
            kernel_size=1)

    def forward(self, x):
        out = self.model(x)
        return out["out"]