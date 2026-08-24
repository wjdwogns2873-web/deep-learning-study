from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import base64
import tempfile
import os
import numpy as np
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import torch
from concurrent.futures import ThreadPoolExecutor

# CPU 집중 작업 및 AI 추론을 처리할 스레드 풀 생성
executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title='Fruit Object Detection AI Server')

app.add_middleware(
    CORSMiddleware, 
    allow_origins=['*'], # 테스트 시 전체 허용
    allow_credentials=True, 
    allow_methods=['*'], 
    allow_headers=['*'],
)

weight_path = 'runs/apple_banana_orange/weights/best.pt'

if not os.path.exists(weight_path):
    weight_path = 'best.pt'

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')
model = YOLO(weight_path).to(device)

# 임시 저장된 비디오 파일 경로를 저장할 변수
current_video_path = None

def process_and_predict(raw_bytes):
    # CPU 바운드 연산 및 YOLO 추론을 담당하는 순수 동기 함수
    # 바이너리 바이트 -> OpenCV Image 디코딩 (base64 과정 생략)
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return None

    # 리사이징 및 YOLO 추론
    resized_frame = cv2.resize(frame, (640, 360))
    results = model(resized_frame, verbose=False)
    annotated_frame = results[0].plot()

    # JPEG 인코딩 후 바이너리 바이트로 변환
    _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    return buffer.tobytes()

@app.websocket('/ws/detect')
async def websocket_detect(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()
    try:
        while True:
            # 프론트엔드로부터 순수 바이너리(bytes) 이미지 수신
            img_bytes = await websocket.receive_bytes()

            # 이벤트 루프가 멈추지 않도록 스레드 풀에서 무거운 연산 수행
            processed_bytes = await loop.run_in_executor(executor, process_and_predict, img_bytes)

            if processed_bytes is None:
                continue

            # 프론트엔드로 순수 바이너리 바이트 즉시 변환
            await websocket.send_bytes(processed_bytes)

    except WebSocketDisconnect:
        print('WebSocket 연결 종료')

@app.get('/')
def read_root():
    return {'status': 'AI Server is running'}

# Form 데이터를 float 타입으로 지정
@app.post('/predict/image')
async def predict(
    file: UploadFile = File(...), 
    conf_value: float = Form(...), 
    iou_value: float = Form(...)
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='이미지 파일만 업로드 가능합니다.')

    # print(f'[predict] type(file): {type(file)}') # <class 'starlette.datastructures.UploadFile'>

    # 수신한 이미지 데이터 읽기
    image_bytes = await file.read()
    # print(f'[predict] type(image_bytes): {type(image_bytes)}') # <class 'bytes'>
    image = Image.open(io.BytesIO(image_bytes))
    # print(f'[predict] type(image): {type(image)}') # <class 'PIL.PngImagePlugin.PngImageFile'>
    # print(f'[predict] type(io.BytesIO(image_bytes)): {type(io.BytesIO(image_bytes))}') # <class '_io.BytesIO'>

    # YOLOv8 추론 실행
    results = model(image, conf=conf_value, iou=iou_value)
    result = results[0]

    detected_objects = []

    # print(f'[DEBUG] 탐지된 객체 수: {len(result.boxes)}')

    # 탐지된 바운딩 박스 / 클래스 / 신뢰도 추출
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = round(float(box.conf[0]), 3)
        xyxy = box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]

        # print(f'[DEBUG] Raw Bounding Box (xyxy): {box.xyxy[0]} | Type: {type(box.xyxy[0])}')
        # tensor([150.5981,  42.6256, 827.5811, 730.3950]) | <class 'torch.Tensor'>
        # print(f'[DEBUG] Confidence Tensor: {box.conf[0]} | Value: {float(box.conf[0]):.3f} | Type: {type(box.conf)}') # <class 'torch.Tensor'>
        # print(f'[DEBUG] Class ID Tensor: {box.cls[0]} | Int Value: {int(box.cls[0])}')
        # print(f'[DEBUG] Model Names Type: {type(model.names)}') # <class 'dict'>

        detected_objects.append({
            'label': class_name, 
            'confidence': confidence, 
            'box': [int(coord) for coord in xyxy]
        })

    # det_obj_for_print = detected_objects[0]

    # print(f"[predict] det_obj_for_print['box']: {det_obj_for_print['box']}") # [116, 62, 798, 752]
    # print(f"[predict] type(det_obj_for_print['box']): {type(det_obj_for_print['box'])}") # <class 'list'>

    # 바운딩 박스가 그려진 이미지 생성 (BGR -> RGB 변환)
    # 박스가 그려진 이미지를 웹으로 보내기 위해 인코딩
    plotted_image = result.plot()
    # print(f'[predict] type(plotted_image): {type(plotted_image)}') # <class 'numpy.ndarray'>
    # print(f'[predict] plotted_image.shape: {plotted_image.shape}') # (1000, 1000, 3)
    _, buffer = cv2.imencode('.jpg', plotted_image)
    # print(f'[predict] type(buffer): {type(buffer)}') # <class 'numpy.ndarray'>
    # print(f'[predict] buffer.shape: {buffer.shape}') # (333519,)

    # 이미지를 base64 이미지로 변환하여 자바 서버로 전달
    base64_image = base64.b64encode(buffer).decode('utf-8')
    # print(f'[predict] type(base64_image): {type(base64_image)}') # <class 'str'>
    

    # 자바 백엔드로 보낼 JSON 응답 생성
    return {
        'status': 'SUCCESS', 
        'total_count': len(detected_objects), 
        'predictions': detected_objects, 
        'result_image_base64': base64_image
    }


# @app.websocket("/ws/detect")
# async def websocket_detect(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             # 1. 프론트엔드로부터 base64 이미지 스트링 수신
#             data = await websocket.receive_text()
            
#             # 헤더 제거 ("data:image/jpeg;base64," 연쇄 분리)
#             if "," in data:
#                 data = data.split(",")[1]
                
#             # Base64 -> OpenCV Image 디코딩
#             img_bytes = base64.b64decode(data)
#             np_arr = np.frombuffer(img_bytes, np.uint8)
#             frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#             if frame is None:
#                 continue

#             # 2. YOLO 추론 (해상도 리사이징으로 처리 속도 극대화)
#             resized_frame = cv2.resize(frame, (640, 360))
#             results = model(resized_frame, verbose=False)
#             annotated_frame = results[0].plot()

#             # 3. 인코딩 및 Base64 변환
#             _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
#             encoded_img = base64.b64encode(buffer).decode('utf-8')

#             # 4. 프론트로 즉시 반환
#             await websocket.send_text(f"data:image/jpeg;base64,{encoded_img}")

#     except WebSocketDisconnect:
#         print("WebSocket 연결 종료")


# @app.post('/upload_video')
# async def upload_video(file: UploadFile = File(...)):
#     global current_video_path

#     # 기존 임시 파일 삭제
#     if current_video_path and os.path.exists(current_video_path):
#         os.remove(current_video_path)

#     # print(f'[upload_video] type(file): {type(file)}') # <class 'starlette.datastructures.UploadFile'>

#     # 업로드된 영상 파일을 임시 폴더에 저장
#     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#     contents = await file.read()
#     temp_file.write(contents)
#     temp_file.close()
#     # print(f'[upload_video] type(contents): {type(contents)}') # <class 'bytes'>
#     # print(f'[upload_video] type(temp_file): {type(temp_file)}') # <class 'tempfile._TemporaryFileWrapper'>

#     current_video_path = temp_file.name
#     # print(f'[upload_video] current_video_path: {current_video_path}')
#     # /var/folders/7q/93pc03s97y9_4w70yrpzflvc0000gn/T/tmp8_i3mnd2.mp4
#     return {'status': 'SUCCESS', 'message': 'Video uploaded successfully'}

# 동영상 프레임 생성 및 보여주기
# def generate_video_frames(video_path):
#     cap = cv2.VideoCapture(video_path)

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             # 동영상이 끝나면 처음부터 무한반복 재생하고싶을 경우
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             continue

#         # verbose=False: 추론 터미널 로그 끄기
#         results = model(frame, verbose=False)
#         annotated_frame = results[0].plot()

#         ret, buffer = cv2.imencode('.jpg', annotated_frame)
#         frame_bytes = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     cap.release()

# async def generate_base64_frames(video_bytes: bytes):
#     with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as temp_video:
#         print('스트리밍 시작..')
#         temp_video.write(video_bytes)
#         temp_video.flush()

#         cap = cv2.VideoCapture(temp_video.name)


#         frame_count = 0
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60] # 품질 60%로 압축

#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 break


#             frame_count += 1
#             if frame_count % 2 != 0: # 2프레임당 1번만 추론/전송
#                 continue

#             # 해상도 축소
#             resized_frame = cv2.resize(frame, (640, 360))
            

#             # YOLO 추론 및 바운딩 박스 그리기 verbose=False: 추론 터미널 로그 끄기
#             results = model(frame, verbose=False)
#             annotated_frame = results[0].plot()

#             # 프레임을 JPEG로 압축 후 base64 변환
#             _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
#             base64_str = base64.b64encode(buffer).decode('utf-8')

#             # SSE(Server-Sent Events) 규격으로 실시간 전송 (\n 필수)
#             yield f'data:image/jpeg;base64,{base64_str}\n\n'

#             # 비동기 이벤트 루프 타임 생성 (버퍼 방출 플러시)
#             await asyncio.sleep(0.01)
#         print('스트리밍 끝..')
#         cap.release()

# @app.post('/stream/video')
# async def detect_video_stream(file: UploadFile = File(...)):
#     video_bytes = await file.read()
#     return StreamingResponse(
#         generate_base64_frames(video_bytes), 
#         media_type='text/event-stream' # 버퍼링을 강제로 끄는 MIME 타입
#     )
            

# @app.get('/video_feed')
# async def video_feed():
#     global current_video_path
#     if not current_video_path or not os.path.exists(current_video_path):
#         raise HTTPException(status_code=400, detail='No video uploaded')

#     return StreamingResponse(
#         generate_video_frames(current_video_path), 
#         media_type='multipart/x-mixed-replace; boundary=frame'
#     )

# @app.post('/stream/video')
# async def detect_video(file: UploadFile = File(...)):
#     # 파일 전체를 메모리에 읽지 않고 임시 파일로 바로 쓰기
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
#         shutil.copyfileobj(file.file, tmp)
#         tmp_path = tmp.name

#     def generate_frames():
#         cap = cv2.VideoCapture(tmp_path)
#         frame_count = 0

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_count += 1
#             if frame_count % 3 != 0: # 3프레임당 1번만 추론
#                 continue

#             # 해상도 줄이기 (속도 향상 핵심)
#             frame = cv2.resize(frame, (480, 270))

#             results = model(frame, verbose=False)
#             annotated = results[0].plot()

#             _, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
#             base64_str = base64.b64encode(buffer).decode('utf-8')

#             # SSE 표준 포맷 전송
#             yield f'data:image/jpeg;base64,{base64_str}\n\n'

#         cap.release()
#         os.remove(tmp_path) # 처리 완료 후 삭제

#     return StreamingResponse(generate_frames(), media_type='text/event-stream')
