import io
import os
# OpenMP 중복 초기화 에러 우회
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms

app = FastAPI()

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
model.eval()

# 이미지 업로드 및 모델 추론
@app.post('/predict/image')
async def predict_image(file: UploadFile = File(...)):
    # 클라이언트가 보낸 파일 읽기
    request_bytes = await file.read()

    # PIL Image를 거쳐 PyTorch Tensor로 변환
    image = Image.open(io.BytesIO(request_bytes)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # 배치 차원 추가 (1, 3, 224, 224)

    # 모델 추론
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred = outputs.max(1)

    # 결과 반환
    return {
        'filename': file.filename, 
        'predicted_class': int(pred.item())
    }
    
import cv2
import numpy as np
from fastapi.responses import StreamingResponse

# 결과 이미지 리턴하기 (바운딩 박스 그리기)
@app.post('/detect/visualize')
async def detect_and_visualize(file: UploadFile = File(...)):
    file_bytes = await file.read()

    # OpenCV 형태로 이미지 디코딩
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 3)
    cv2.putText(img, 'Object: 98%', (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 이미지를 메모리상에서 바로 JPEG 바이너리로 인코딩
    _, encoded_img = cv2.imencode('.jpg', img)
    io_buf = io.BytesIO(encoded_img.tobytes())

    return StreamingResponse(io_buf, media_type='image/jpeg')

def generate_video_frames():
    camera = cv2.VideoCapture(0)