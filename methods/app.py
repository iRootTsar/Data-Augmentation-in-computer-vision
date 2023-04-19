import gradio as gr
import torch
import numpy as np
from PIL import Image
import nnmodel
from pathlib import Path
import torchvision.transforms as transforms

# Define the same transformations as used in training
transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model():
    model = nnmodel.SymmetricNet2(0.5)
    model_path = Path(__file__).resolve().parent / "best_model.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model().to(device)

def predict(image: Image.Image):
    # Preprocess the image
    input_image = transform(image).unsqueeze(0).to(device)

    # Make a prediction using the trained model
    model.eval()
    with torch.no_grad():
        outputs = model(input_image)
        _, predicted = torch.max(outputs.data, 1)

    # Return the result
    label_map = {0: "Sphaleron", 1: "Black Hole"}
    return label_map[predicted.item()]


image_input = gr.inputs.Image(shape=(50, 50))
label_output = gr.outputs.Label()

gr.Interface(fn=predict, inputs=image_input, outputs=label_output).launch()
