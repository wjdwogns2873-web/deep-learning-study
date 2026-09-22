import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
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

model_dir = 'runs/3fruits/weights'
if not os.path.exists(model_dir):
    model_dir = ''

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'device: {device}', flush=True)

pt_path = os.path.join(model_dir, 'best.pt')
yolo_model = YOLO(pt_path) if os.path.exists(pt_path) else None

available_providers = ort.get_available_providers()
providers = [p for p in ['CoreMLExecutionProvider', 'CPUExecutionProvider'] if p in available_providers]

fp32_path = os.path.join(model_dir, 'best.FP32.onnx')
onnx_fp32_session = ort.InferenceSession(fp32_path, providers=providers) if os.path.exists(fp32_path) else None

fp16_path = os.path.join(model_dir, 'best.FP16.onnx')
onnx_fp16_session = ort.InferenceSession(fp16_path, providers=providers) if os.path.exists(fp16_path) else None

executor = ThreadPoolExecutor(max_workers=4)

class_dict = {0: 'apple', 1: 'banana', 2: 'orange'}
colors = {0: (0, 0, 255), 1: (0, 255, 255), 2: (0, 165, 255)}

def process_and_predict(raw_bytes: bytes, 
                        model_type: str = 'FP16', 
                        conf_threshold: float = 0.25, 
                        iou_threshold: float = 0.45) -> bytes:
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    h, w, _ = frame.shape

    # 1. PyTorch (.pt) 추론
    if model_type == 'PyTorch' and yolo_model:
        results = yolo_model(frame, conf=conf_threshold, iou=iou_threshold, device=device, verbose=False)
        plotted_frame = results[0].plot()
        _, buffer = cv2.imencode('.jpg', plotted_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        return buffer.tobytes()

    # 2. ONNX (FP16 / FP32) 추론
    session = onnx_fp16_session if model_type == 'FP16' and onnx_fp16_session else onnx_fp32_session
    if not session:
        session = onnx_fp32_session

    if not session:
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    img_resized = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # ONNX 입력 텐서 준비
    is_fp16 = (model_type == 'FP16')
    tensor_type = np.float16 if is_fp16 else np.float32
    input_tensor = img_rgb.transpose(2, 0, 1).astype(tensor_type) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    try:
        outputs = session.run([output_name], {input_name: input_tensor})
    except Exception:
        # FP16 실행 실패 시 FP32 세션 fallback
        outputs = onnx_fp32_session.run([output_name], {input_name: input_tensor.astype(np.float32)})

    predictions = np.squeeze(outputs[0]).astype(np.float32) # (7, 8400)

    boxes_raw = predictions[:4, :] # (4, 8400)
    scores_raw = np.max(predictions[4:, :], axis=0) # (8400,)
    class_ids_raw = np.argmax(predictions[4:, :], axis=0) # (8400,)

    mask = scores_raw > conf_threshold
    boxes_filtered = boxes_raw[:, mask]
    scores_filtered = scores_raw[mask]
    class_ids_filtered = class_ids_raw[mask]

    if len(scores_filtered) > 0:
        # NMS를 위한 Bounding Box 변환 [x, y, w, h] (640 기준)
        cx, cy, bw, bh = boxes_filtered[0], boxes_filtered[1], boxes_filtered[2], boxes_filtered[3]
        x = (cx - bw / 2) * (w / 640)
        y = (cy - bh / 2) * (h / 640)
        w_box = bw * (w / 640)
        h_box = bh * (h / 640)

        boxes_nms = np.vstack((x, y, w_box, h_box)).T.tolist()
        scores_nms = scores_filtered.tolist()

        # Fast NMS 적용 (박스 중복 제거)
        indices = cv2.dnn.NMSBoxes(boxes_nms, scores_nms, conf_threshold, iou_threshold)

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes_nms[i]
                xmin, ymin, bw_i, bh_i = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                xmax, ymax = xmin + bw_i, ymin + bh_i

                cls_id = int(class_ids_filtered[i])
                score = scores_filtered[i]
                label = f"{class_dict.get(cls_id, 'fruit')} {score:.2f}"
                color = colors.get(cls_id, (0, 255, 0))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, label, (xmin, max(ymin - 10, 15)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
    return buffer.tobytes()

@app.get('/')
def read_root():
    return {'status': 'AI Server is Running'}

@app.websocket('/ws/detect')
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    frame_queue = asyncio.Queue(maxsize=1) # 큐 크기 1로 줄여 최신 프레임만 처리
    loop = asyncio.get_running_loop()

    async def consumer():
        try:
            while True:
                item = await frame_queue.get()
                # print(f'item: {item}', flush=True)
                img_bytes, model_type = item['bytes'], item['model_type']

                processed_bytes = await loop.run_in_executor(
                    executor, process_and_predict, img_bytes, model_type, 0.25, 0.45
                )

                if processed_bytes is not None:
                    await websocket.send_bytes(processed_bytes)

                frame_queue.task_done()
        except asyncio.CancelledError:
            pass

    consumer_task = asyncio.create_task(consumer())

    try:
        while True:
            raw_data = await websocket.receive_bytes()
            # print(f'raw_data: {raw_data}', flush=True)

            if len(raw_data) > 4:
                meta_len = int.from_bytes(raw_data[:4], byteorder='big')
                meta_json = json.loads(raw_data[4:4 + meta_len].decode('utf-8'))
                img_bytes = raw_data[4 + meta_len:]
                model_type = meta_json.get('model_type', 'FP16')
            else:
                img_bytes = raw_data
                model_type = 'FP16'

            # 오래된 프레임 버리고 항상 최신 프레임 채우기
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                    frame_queue.task_done()
                except asyncio.QueueEmpty:
                    pass

            await frame_queue.put({'bytes': img_bytes, 'model_type': model_type})

    except WebSocketDisconnect:
        pass
    finally:
        consumer_task.cancel()
        await asyncio.gather(consumer_task, return_exceptions=True)