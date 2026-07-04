import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI, UploadFile, File
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import io
from PIL import Image
import torchvision.models as models

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
# model = SimpleCNN().to(device)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 6)

model = model.to(device)

# model.load_state_dict(torch.load('my_custom_cnn_fruits.pth', map_location=device))
model.load_state_dict(torch.load('fold1_100.pth', map_location=device))
model.eval()

transform = A.Compose([
    A.Resize(128, 128), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2()
])

class_names = ["fresh_apple", "rotten_apple", "fresh_banana", "rotten_banana", "fresh_orange", "rotten_orange"]

@app.post("/predict")
async def predict_fruits(file: UploadFile = File(...)):
    contents = await file.read()

    pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = np.array(pil_image)
    # image_cv = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    aug = transform(image=image)
    input_tensor = aug['image'].unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor.to(device))

        probs = torch.softmax(outputs, dim=1)
        # probs = torch.sigmoid(outputs)
        
        pred = outputs.argmax(1).item()

        confidence = probs[0][pred].item() * 100

    prediction = class_names[pred]

    top2_values, top2_indices = torch.topk(probs, k=2, dim=1)
    # print(f"******top2_values: {top2_values}")
    # print(f"******top2_indices: {top2_indices}")
    second_prob = top2_values[0][1].item() * 100
    second_index = top2_indices[0][1].item()

    # print(f"******second_prob: {second_prob}")
    # print(f"******second_index: {second_index}")
    # print(f"******second_prob type: {type(second_prob)}")
    
    second_predicton = class_names[second_index]

    return {
        "prediction": prediction, 
        "confidence": f"{confidence:.2f}", 
        "second_predicton": second_predicton, 
        "second_prob": f"{second_prob:.2f}"
        # "all_probs": probs[0].tolist()
    }