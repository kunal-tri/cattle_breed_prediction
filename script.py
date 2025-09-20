import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys

# Updated class_labels list
class_labels = [
    "Brown_Swiss", "Gir", "Hariana", "Holstein_Friesian", "Jaffrabadi",
    "Jersey", "Kankrej", "Mehsana", "Murrah", "Nili_Ravi", "Ongole",
    "Rathi", "Red_Sindhi", "Sahiwal", "Surti", "Tharparkar"
]

def load_model(model_path, device):
    # Try loading as TorchScript first
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        return model
    except Exception:
        # Fallback to standard PyTorch model
        model = torch.load(model_path, map_location=device)
        if isinstance(model, nn.Module):
            model.eval()
            return model
        else:
            raise ValueError("The .pt file does not contain a torch.nn.Module or TorchScript model")

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, predicted = probs.max(1)
        class_idx = predicted.item()
        confidence = max_prob.item()
        
        # Check if confidence is less than 40%
        if confidence < 0.40:
            return "Wrong image (Low confidence)"
        
        if 0 <= class_idx < len(class_labels):
            return f"{class_labels[class_idx]} (Confidence: {confidence:.2f})"
        else:
            return f"Unknown breed (Class index {class_idx})"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py model.pt input.jpg")
        sys.exit(1)

    model_path, image_path = sys.argv[1], sys.argv[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device)
    image_tensor = preprocess_image(image_path)

    prediction = predict(model, image_tensor, device)
    print("Predicted cattle breed:", prediction)
