import torchvision.models.segmentation as segm

model = segm.deeplabv3_resnet101(weights="DEFAULT")