import torch
import torch.nn as nn
import torchvision.models.segmentation as segm


class Model(nn.Module):
    def __init__(self, n_classes=19, pretrained_path="resnet101-63fe2227.pth"):
        super().__init__()

        self.model = segm.deeplabv3_resnet101(
            weights=None,
            aux_loss=True
        )

        state_dict = torch.load(pretrained_path, map_location="cpu")
        self.model.backbone.load_state_dict(state_dict, strict=False)

        self.model.classifier[-1] = nn.Conv2d(
            in_channels=256,
            out_channels=n_classes,
            kernel_size=1
        )

        if self.model.aux_classifier is not None:
            self.model.aux_classifier[-1] = nn.Conv2d(
                in_channels=1024,
                out_channels=n_classes,
                kernel_size=1
            )

    def forward(self, x):
        out = self.model(x)
        return out["out"], out["aux"]