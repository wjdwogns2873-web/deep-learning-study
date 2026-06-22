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

app = FastAPI()

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__() # (32, 3, 128, 128) -> (32, 16, 64, 64) -> (32, 32, 32, 32) -> (32, 64, 16, 16)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64 * 16 * 16, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('simple_cnn_apple.pth', map_location=device))
model.eval()

transform = A.Compose([
    A.Resize(128, 128), 
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
    ToTensorV2()
])

class_names = ['fresh', 'rotten']

@app.post("/predict")
async def predict_apple(file: UploadFile = File(...)):
    # 1. 사용자가 보낸 파일 바이트를 읽어옵니다.
    contents = await file.read()

    # 2. 바이트 데이터를 OpenCV나 PIL이 읽을 수 있는 이미지 배열로 변환합니다.
    pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
    image_np = np.array(pil_image) # Albumentations는 Numpy배열을 받습니다.
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # OpenCV 스타일로 변환

    transformed = transform(image=image_cv)
    input_tensor = transformed['image'].unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        pred = outputs.argmax(1)

    result_string = class_names[pred]

    return {"prediction": result_string}