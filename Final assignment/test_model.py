import torch
from model import Model

device = "cuda" if torch.cuda.is_available() else "cpu"

# Maak model
model = Model(in_channels=3, n_classes=19).to(device)
model.eval()

# Fake input (zoals Cityscapes)
x = torch.randn(1, 3, 512, 1024).to(device)

with torch.no_grad():
    y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)