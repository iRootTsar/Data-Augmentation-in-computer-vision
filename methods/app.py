import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append("Data-Augmentation-in-computer-vision/methods")

import nnmodel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nnmodel.VGGNet2(0.5).to(device)
model_path = str(Path.cwd() / "methods" / "models" / "best_model3.pth")

model.load_state_dict(torch.load(model_path, map_location=device))

def preprocess_image(input_image: np.ndarray):
    input_image = np.expand_dims(input_image, axis=0)
    input_image = np.moveaxis(input_image, -1, 1)
    img_tensor = torch.from_numpy(input_image).float().to(device)
    return img_tensor

def predict(input_image: Image.Image):
    img_array = np.array(input_image)

    img_tensor = preprocess_image(img_array)
    with torch.no_grad():
        output = model(img_tensor)
        _, prediction = torch.max(output, 1)

    class_label = "Black Hole" if prediction.item() == 0 else "Sphaleron"
    return class_label

image_input = gr.inputs.Image(label="Select an image")
label_output = gr.outputs.Label(num_top_classes=1)

iface = gr.Interface(fn=predict, inputs=image_input, outputs=label_output, title="Image Classification",
                     description="Upload a black hole or sphaleron image and the classifier will predict the class.")
iface.launch()
