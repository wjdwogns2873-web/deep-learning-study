import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, unquote
import json
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import torch

app = FastAPI(title='Fruit Object Detection AI Server')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드 및 추론 세션 초기화
model_dir = 'runs/3fruits/weights'

if not os.path.exists(model_dir):
    model_dir = ''

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'device: {device}')

pt_path = os.path.join(model_dir, 'best.pt')
yolo_model = YOLO(pt_path)

available_providers = ort.get_available_providers()
providers = [p for p in ['CoreMLExecutionProvider', 'CPUExecutionProvider'] if p in available_providers]

# ONNX FP32
fp32_path = os.path.join(model_dir, 'best.FP32.onnx')
onnx_fp32_session = ort.InferenceSession(fp32_path, providers=providers)

# ONNX FP16
fp16_path = os.path.join(model_dir, 'best.FP16.onnx')
onnx_fp16_session = ort.InferenceSession(fp16_path, providers=providers)

# 스레드 풀 생성 (CPU 바운드 추론 연산 적용)
executor = ThreadPoolExecutor(max_workers=4)

class_dict = {0: 'apple', 1: 'banana', 2: 'orange'}
colors = {0: (0, 0, 255), 1: (0, 255, 255), 2: (0, 165, 255)}

# ONNX & PyTorch 공통 추론 함수
def process_and_predict(raw_bytes: bytes, 
                        model_type: str = 'FP16', 
                        conf_threshold: float = 0.5, 
                        iou_threshold: float = 0.5) -> bytes:
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    h, w, _ = frame.shape

    # PyTorch (.pt) 추론
    if model_type == 'PyTorch':
        results = yolo_model(frame, conf=conf_threshold, iou=iou_threshold, device=device, verbose=False)
        plotted_frame = results[0].plot()
        _, buffer = cv2.imencode('.jpg', plotted_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        return buffer.tobytes()

    # ONNX (FP16 / FP32) 추론
    session = onnx_fp16_session if model_type == 'FP16' and onnx_fp16_session else onnx_fp32_session
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    img_resized = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_tensor = img_rgb.transpose(2, 0, 1).astype(np.float16 if model_type == 'FP16' else np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    outputs = session.run([output_name], {input_name: input_tensor})
    # print(f'type(outputs): {type(outputs)}') # <class 'list'>
    # print(f'outputs[0].shape: {outputs[0].shape}') # (1, 7, 8400)
    predictions = np.squeeze(outputs[0])
    # print(f'predictions.shape: {predictions.shape}') # (7, 8400)

    boxes = predictions[:4, :]
    # print(f'boxes.shape: {boxes.shape}')
    scores = np.max(predictions[4:, :], axis=0)
    # print(f'scores.shape: {scores.shape}') # (8400,)
    class_ids = np.argmax(predictions[4:, :], axis=0)
    # print(f'class_ids.shape: {class_ids.shape}') # (8400,)

    valid_indices = scores > conf_threshold
    # print(f'valid_indices.shape: {valid_indices.shape}')
    final_boxes = boxes[:, valid_indices]
    # print(f'final_boxes.shape: {final_boxes.shape}')
    final_scores = scores[valid_indices]
    # print(f'final_scores.shape: {final_scores.shape}')
    final_class_ids = class_ids[valid_indices]
    # print(f'final_class_ids.shape: {final_class_ids.shape}')

    for i in range(final_boxes.shape[1]):
        cx, cy, bw, bh = final_boxes[:, i]
        xmin = int( (cx - bw / 2) * (w / 640) )
        ymin = int( (cy - bh / 2) * (h / 640) )
        xmax = int( (cx + bw / 2) * (w / 640) )
        ymax = int( (cy + bh / 2) * (h / 640) )

        cls_id = int(final_class_ids[i])
        label = f"{class_dict.get(cls_id, 'fruit')} {final_scores[i]:.2f}"

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), colors.get(cls_id, (0, 255, 0)), 2)
        cv2.putText(frame, label, (xmin, ymin-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors.get(cls_id, (0, 255, 0)), 2)

    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    return buffer.tobytes()

@app.get('/')
def read_root():
    return {'status': 'AI Server is Running'}

@app.websocket('/ws/detect')
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    frame_queue = asyncio.Queue(maxsize=2)
    loop = asyncio.get_running_loop()

    async def consumer():
        try:
            while True:
                item = await frame_queue.get()
                img_bytes, model_type = item['bytes'], item['model_type']

                processed_bytes = await loop.run_in_executor(executor, process_and_predict, 
                                                             img_bytes, model_type, 0.25, 0.45)

                if processed_bytes is not None:
                    await websocket.send_bytes(processed_bytes)

                frame_queue.task_done()
        except asyncio.CancelledError:
            pass

    consumer_task = asyncio.create_task(consumer())

    try:
        while True:
            raw_data = await websocket.receive_bytes()

            # 헤더 메타데이터 분리 (Length Prefix Parsing)
            if len(raw_data) > 4:
                meta_len = int.from_bytes(raw_data[:4], byteorder='big')
                meta_json = json.loads(raw_data[4:4 + meta_len].decode('utf-8'))
                img_bytes = raw_data[4 + meta_len:]
                model_type = meta_json.get('model_type', 'FP16')
            else:
                img_bytes = raw_data
                model_type = 'FP16'

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                    frame_queue.task_done()
                except asyncio.QueueEmpty:
                    pass

            await frame_queue.put({'bytes': img_bytes, 'model_type': model_type})

    except WebSocketDisconnect:
        print('WebSocket 연결 종료')
    finally:
        consumer_task.cancel()
        await asyncio.gather(consumer_task, return_exceptions=True)

@app.post('/predict/image')
async def predict(file: UploadFile = File(...), 
                  conf_value: float = Form(...), 
                  iou_value: float = Form(...), 
                  model_type: str = Form('FP16')):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='이미지 파일만 업로드 가능합니다.')

    image_bytes = await file.read()
    loop = asyncio.get_running_loop()

    # 스레드 풀에서 추론 연산 실행
    processed_bytes = await loop.run_in_executor(executor, process_and_predict, 
                                                 image_bytes, model_type, conf_value, iou_value)

    meta_data = {'status': 'SUCCESS', 'engine': model_type}
    json_str = quote(json.dumps(meta_data, ensure_ascii=False))

    return Response(
        content=processed_bytes, 
        media_type='image/jpeg', 
        headers={
            'X-Detection-Meta': json_str
        }
    )