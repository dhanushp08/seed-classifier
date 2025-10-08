import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import timm

# === CONFIGURATION ===
model_path = "xception_epoch2.pth"  # 👈 your model checkpoint
class_names = ['Alfalfa', 'Broccoli', 'Chick Pea', 'Clover', 'Green Peas', 'Mung Bean', 'Radish']  

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Xception expects 299x299
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === LOAD MODEL ===
model = timm.create_model('xception', pretrained=False, num_classes=len(class_names))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# === PREDICT FUNCTION ===
def predict(image):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        outputs = model(img_tensor)
        pred = torch.argmax(outputs, 1).item()
        return f"Predicted Seed Type: {class_names[pred]}"

# === GRADIO INTERFACE ===
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🌿 Xception Seed Classifier",
    description="Upload a seed image and get the predicted seed type."
).launch()
