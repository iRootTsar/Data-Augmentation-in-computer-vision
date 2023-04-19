import gradio as gr
import torch
import numpy as np
from dataloader import *
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import os
from dataloader import *

import sys
from pathlib import Path
import nnmodel

# Constructs a path to a directory that contains dataloader.py and plotCreator.py
module_path = str(Path.cwd().parents[0] / "methods")

# Checks to see if the directory is already in sys.path to avoid adding it multiple times.
if module_path not in sys.path:
    sys.path.append(module_path)
    
# Creates two file paths pointing to two HDF5 files
data_path0 = str(Path.cwd().parent / "data" / "BH_n4_M10_res50_15000_events.h5")
data_path1 = str(Path.cwd().parent / "data" / "PP13-Sphaleron-THR9-FRZ15-NB0-NSUBPALL_res50_15000_events.h5")


# Reads the two HDF5 data files and creates two NumPy arrays
bhArray = dataToArray(data_path0)
sphArray = dataToArray(data_path1)

# Define the possible image choices for the dropdown
image_choices = [f"BH Image {i}" for i in range(len(bhArray))] + [f"Sphaleron Image {i}" for i in range(len(sphArray))]

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nnmodel.SymmetricNet2(0.5).to(device)
model_path = os.path.join(os.getcwd(), "models", "best_model.pth")

model.load_state_dict(torch.load(model_path, map_location=device))

def preprocess_image(input_image: Image.Image):
    transforms = Compose([
        ToTensor()
    ])
    img_tensor = transforms(input_image)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    return img_tensor

def predict(input_image: str):
    if input_image.startswith("BH"):
        img_array = bhArray[int(input_image.split(" ")[-1])]
    else:
        img_array = sphArray[int(input_image.split(" ")[-1])]
    
    # Convert the image array to a PIL Image object
    img = Image.fromarray(img_array)
    
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        output = model(img_tensor)
        _, prediction = torch.max(output, 1)
    
    class_label = "Black Hole" if prediction.item() == 0 else "Sphaleron"
    return class_label

# Define the input dropdown and output label
image_input = gr.inputs.Dropdown(choices=image_choices, label="Select an image")
label_output = gr.outputs.Label(num_top_classes=1)

# Create the interface
iface = gr.Interface(fn=predict, inputs=image_input, outputs=label_output, title="Image Classification", 
                     description="Select an image from the dropdown menu and the classifier will predict whether it is a black hole or a sphaleron.")
iface.launch()
