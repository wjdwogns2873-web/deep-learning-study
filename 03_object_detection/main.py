from ultralytics import YOLO
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
import io
from PIL import Image

app = FastAPI()

# 학습 완료된 리즈시절 가중치 가져오기
model = YOLO('runs/trash_v1/weights/best.pt')

@app.post("/predict")
async def predict_trash(file: UploadFile):
    # 업로드된 파일 읽어서 OpenCV 이미지 객체로 변환하기
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
    # image_rgb = np.array(pil_image)
    # img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # YOLOv8로 이미지 추론하기
    # conf 값을 아주 낮게 주어서 소심한 예측 박스도 보여준다.
    results = model(img, conf=0.01)

    # Ultralytics는 결과 이미지에 자동으로 박스를 그려주는 내장 함수를 제공한다.
    # results[0].plot()을 호출하면 네모박스와 라벨이 그려진 numpy 배열 이미지가 나온다.
    annotated_img = results[0].plot()

    # 박스가 그려진 이미지를 다시 웹으로 전송하기 위해 인코딩하기
    _, im_png = cv2.imencode(".png", annotated_img)

    # StreamingResponse를 사용하면 브라우저에 바로 이미지를 쏴줄 수 있다.
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    