from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
import io
import cv2
import numpy as np
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates2")

model = YOLO("runs/apple_prediction/weights/best.pt")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

@app.post("/predict")
async def predict_apple(file: UploadFile = File(...)):
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(source=img, conf=0.4, iou=0.7, device="mps")

    annotated_img = results[0].plot()

    _, im_png = cv2.imencode(".png", annotated_img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    