import torch
import torch.nn as nn
import torchvision.models.segmentation as segm

class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19, pretrained=True):
        super().__init__()
        # load the bakcbone model
        self.model = segm.deeplabv3_resnet50(pretrained=pretrained, progress=True)
        #self.model = segm.deeplabv3_resnet101(pretrained=pretrained, progress=True)
        #self.model = segm.deeplabv3_mobilenet_v3_large(pretrained=pretrained, progress=True)        

        self.model.classifier[-1] = nn.Conv2d(256, n_classes, kernel_size=1)
        
        # low-level features
        if self.model.aux_classifier is not None:
            self.model.aux_classifier[-1] = nn.Conv2d(256, n_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']


