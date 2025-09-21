from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Load TorchScript model (make sure you committed model.pt)
model = torch.jit.load("best_convnext_large_scripted.pt", map_location="cpu")
model.eval()

# TODO: Replace this with the exact list of your cattle breeds
CLASSES = [
    "Brown_Swiss",
        "Gir",
        "Hariana",
        "Holstein_Friesian",
        "Jaffrabadi",
        "Jersey",
        "Kankrej",
        "Mehsana",
        "Murrah",
        "Nili_Ravi",
        "Ongole",
        "Rathi",
        "Red_Sindhi",
        "Sahiwal",
        "Surti",
        "Tharparkar"
]

# Same preprocessing as training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

app = FastAPI()

def prepare_image(file):
    image = Image.open(file).convert("RGB")
    return preprocess(image).unsqueeze(0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    tensor = prepare_image(file.file)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        idx = probs.argmax(dim=1).item()
        return {
            "predicted_class": CLASSES[idx],
            "probability": float(probs[0][idx])
        }
