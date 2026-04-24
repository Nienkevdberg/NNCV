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

        
        self.model.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, n_classes, kernel_size=1),
        )

        if self.model.aux_classifier is not None:
            self.model.aux_classifier = nn.Conv2d(
                in_channels=256,
                out_channels=n_classes,
                kernel_size=1
            )

    def forward(self, x):
        out = self.model(x)
        if "aux" in out:
            return out["out"], out["aux"]
        return out["out"]
