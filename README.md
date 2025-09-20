
# Cattle Breed Image Classification  

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)  
![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)  
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  

## Overview  
This project builds a deep learning model to classify **Indian cattle breeds** using transfer learning with `timm` pretrained architectures. The notebook is designed for Google Colab but works locally with minor changes.  

## Features  
- Preprocessing and augmentation with `torchvision.transforms`  
- Dataset loading using `torchvision.datasets.ImageFolder`  
- Train/validation/test split  
- Model training with PyTorch and `timm`  
- Evaluation with accuracy and loss metrics  
- Export support to **ONNX**  

## Requirements  
Install dependencies:  
```bash
pip install torch torchvision timm onnx ml_dtypes tqdm matplotlib pillow
```  

## Dataset  
- Place dataset in Google Drive or local path.  
- Directory structure:  
```
Indian_bovine_breeds/
├── Class1/
├── Class2/
├── Class3/
├── ...
```
- Each subfolder = one breed class.  

## Training  
Run the notebook step by step. Key steps:  
1. Mount dataset (e.g., Google Drive).  
2. Define transforms for train/val/test.  
3. Split dataset into 70% train, 20% validation, 10% test.  
4. Train the model:  
   - Optimizer: Adam/SGD  
   - Loss: CrossEntropyLoss  
   - Metrics: Accuracy  
5. Evaluate on validation and test sets.  

## Export Model  
You can export to **ONNX** for deployment:  
```python
torch.onnx.export(model, sample_input, "cattle_breed_classifier.onnx")
```  

## Example Inference  
```python
from PIL import Image
import torchvision.transforms as transforms
import torch

# Load model
model.eval()
image = Image.open("sample.jpg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    pred = output.argmax(1).item()
print("Predicted class:", pred)
```  

## Results  
- Trained on multiple cattle breeds.  
- Accuracy depends on dataset size and augmentation.  

## License  
This repository is under the MIT License.  
