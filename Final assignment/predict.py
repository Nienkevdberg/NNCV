"""
This script provides and example implementation of a prediction pipeline 
for a PyTorch U-Net model. It loads a pre-trained model, processes input 
images, and saves the predicted segmentation masks. 

You can use this file for submissions to the Challenge server. Customize 
the `preprocess` and `postprocess` functions to fit your model's input 
and output requirements.
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose, 
    ToImage, 
    Resize, 
    ToDtype, 
    Normalize,
    InterpolationMode,
)

from PP_model import Model

# Fixed paths inside participant container
# Do NOT chnage the paths, these are fixed locations where the server will 
# provide input data and expect output data.
# Only for local testing, you can change these paths to point to your local data and output folders.
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

def preprocess(img: Image.Image) -> torch.Tensor:
    # Implement your preprocessing steps here
    # For example, resizing, normalization, etc.
    # Return a tensor suitable for model input
    transform = Compose([
        ToImage(),
        Resize(size=(384,768), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    # Implement your postprocessing steps here
    # For example, resizing back to original shape, converting to color mask, etc.
    # Return a numpy array suitable for saving as an image
    pred_max = torch.argmax(pred, dim=1, keepdim=True)
    prediction = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()  # Remove batch and channel dimensions if necessary

    return prediction_numpy

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    model = Model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    image_files = list(Path(IMAGE_DIR).glob("*.png"))  # DO NOT CHANGE, IMAGES WILL BE PROVIDED IN THIS FORMAT
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = np.array(img).shape[:2]

            # Preprocess
            img_tensor = preprocess(img).to(device)

            # flipped input
            img_flip = torch.flip(img_tensor, dims=[3])
            out1 = model(img_tensor)
            out2 = model(img_flip)

            if isinstance(out1, tuple):
                out1 = out1[0]
            if isinstance(out2, tuple):
                out2 = out2[0]

            out2 = torch.flip(out2, dims=[3])
            pred = (out1 + out2) / 2

            # Postprocess to segmentation mask
            seg_pred = postprocess(pred, original_shape)

            # Create mirrored output folder
            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save predicted mask
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)

if __name__ == "__main__":
    main()


