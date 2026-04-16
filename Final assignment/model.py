import torch
import torch.nn as nn
import torchvision.models.segmentation as segm


class Model(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()

        self.model = segm.deeplabv3_resnet101(
            weights=None,
            weights_backbone=None,
            aux_loss=False
        )

        self.model.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=n_classes,
            kernel_size=1
        )

        if self.model.aux_classifier is not None:
            self.model.aux_classifier[4] = nn.Conv2d(
                in_channels=256,
                out_channels=n_classes,
                kernel_size=1
            )

    def forward(self, x):
        return self.model(x)["out"]