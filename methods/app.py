import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
import sys
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append("Data-Augmentation-in-computer-vision/methods")
import dataloader
import nnmodel

os.chdir(os.path.join(os.getcwd(), "Data-Augmentation-in-computer-vision"))
data_path0 = str(Path.cwd() / "data" / "BH_n4_M10_res50_15000_events.h5")
data_path1 = str(Path.cwd() / "data" / "PP13-Sphaleron-THR9-FRZ15-NB0-NSUBPALL_res50_15000_events.h5")

def get2dHistograms(path):
    f = h5py.File(path)
    keys = list(f.keys())
    dataset = [f[key]["data"] for key in keys]
    return dataset

def dataToArray(path):
    return np.array(get2dHistograms(path))

bhArray = dataToArray(data_path0)
sphArray = dataToArray(data_path1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nnmodel.SymmetricNet2(0.5).to(device)
model_path = str(Path.cwd() / "methods" / "models" / "best_model2.pth")

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

def plot_random_images():
    random_bh_indices = np.random.choice(range(len(bhArray)), 5)
    random_sph_indices = np.random.choice(range(len(sphArray)), 5)

    bh_sample = bhArray[random_bh_indices]
    sph_sample = sphArray[random_sph_indices]

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(16, 6))
    for i in range(5):
        ax = axes[i]
        ax.imshow(bh_sample[i], cmap='jet', vmin=0, vmax=255)
        ax.axis('off')
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(16, 6))
    for i in range(5):
        ax = axes[i]
        ax.imshow(sph_sample[i], cmap='jet', vmin=0, vmax=255)
        ax.axis('off')
    plt.show()

plot_random_images()

image_input = gr.inputs.Image(label="Select an image")
label_output = gr.outputs.Label(num_top_classes=1)

iface = gr.Interface(fn=predict, inputs=image_input, outputs=label_output, title="Image Classification",
                     description="Upload a black hole or sphaleron image and the classifier will predict the class.")
iface.launch()
