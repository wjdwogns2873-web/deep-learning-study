from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
import io
import cv2
import numpy as np
from fastapi.templating import Jinja2Templates
import json

app = FastAPI()

templates = Jinja2Templates(directory="templates2")

# model = YOLO("runs/apple_prediction/weights/best.pt")
model = YOLO("runs/apple_banana_org/weights/best.pt")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

@app.post("/predict")
async def predict_fruit(file: UploadFile = File(...)):
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(source=img, conf=0.4, iou=0.4, device="mps")

    # results[0].boxes.cls에 탐지된 객체들의 클래스 번호(0, 1 등)가 텐서 형태로 들어있습니다.
    # results[0].names에는 {0: 'apple', 1: 'banana'} 형태의 사전(dict)이 들어있습니다.
    
    detected_classes = results[0].boxes.cls.cpu().numpy()
    names_dict = results[0].names
    
    print(detected_classes) # [          0]
    print(f"shape: {detected_classes.shape}") # shape: (1,)

    print(names_dict) # {0: 'apples', 1: 'banana'}



    

    counts = {"apples": 0, "banana": 0}
    for cls_id in detected_classes:
        class_name = names_dict[int(cls_id)]
        if class_name in counts:
            counts[class_name] += 1
    
    annotated_img = results[0].plot()
    _, im_png = cv2.imencode(".png", annotated_img)

    # 파이썬 딕셔너리를 글자(JSON)로 변환해서 헤더에 'x-detected-counts'라는 이름으로 담습니다.
    headers = {"x-detected-counts": json.dumps(counts)}
    
    
    return StreamingResponse(
        io.BytesIO(im_png.tobytes()), 
        media_type="image/png", 
        headers=headers # 헤더 실어 보내기
    )
    