# from ultralytics import YOLO
import onnxruntime as ort
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
import io
from PIL import Image

app = FastAPI()

weight_path = 'onnx/vehicle_detection_best.onnx'

@app.post('/predict')
async def predict(file: UploadFile):
    # ONNX 추론 세션 생성
    session = ort.InferenceSession(weight_path)
    
    # 모델이 요구하는 입력(Input) 및 출력(Output) 노드 이름 미리 파악해두기
    input_name = session.get_inputs()[0].name # 보통 'images'
    output_name = session.get_outputs()[0].name # 보통 'output0'

    # OpenCV로 읽은 이미지(img)는 (Height, Width, 3) 형태의 BGR 정수 배열입니다.
    # 하지만 YOLO ONNX 모델은 (1, 3, 640, 640) 형태의 0.0~1.0 실수(float32) 텐서를 원합니다.
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 640 640 리사이즈
    resized = cv2.resize(img, (640, 640))

    # BGR to RGB 색상 변환
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # (640, 640, 3) -> (3, 640, 640)
    transposed = np.transpose(rgb, (2, 0, 1))

    # 정규화
    normalized = transposed.astype(np.float32) / 255.0

    # 맨 앞에 배치 차원 1 추가 -> (1, 3, 640, 640)
    input_tensor = np.expand_dims(normalized, axis=0)

    # ONNX 세션에 입력 텐서를 넣고 결과 받아오기
    outputs = session.run([output_name], {input_name: input_tensor})

    # 결과 배열 꺼내기
    output = outputs[0]

    print(f'output.shape: {output.shape}') # (1, 5, 8400)

    # (1, 5, 8400) -> (5, 8400) -> (8400, 5)
    predictions = outputs[0][0].T

    # 8400개 행을 하나씩 돌면서 conf 점수가 0.5 이상인 것만 필터링
    boxes = []
    confidences = []

    # 원본 이미지의 가로, 세로 크기 (좌표 비율 복원을 위해 필요)
    h_orig, w_orig = img.shape[:2]
    scale_x = w_orig / 640.0
    scale_y = h_orig / 640.0

    # 지금 pred에 있는 픽셀 정보는.. 640 / w를 곱했고, 640 / h를 곱한 상태이다. 그럼 이것의 역수를 곱하면 다시 원래대로 돌아온다.
    
    for pred in predictions:
        conf = float(pred[4])
        if conf > 0.5:
            cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]

            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            bw_scaled = int(bw * scale_x)
            bh_scaled = int(bh * scale_y)

            boxes.append([x1, y1, bw_scaled, bh_scaled])
            confidences.append(conf)

    # OpenCV NMS 실행
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.45)

    # 살아남은 박스들에만 초록색 사각형과 확신도 텍스트 그리기
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            score = confidences[i]

            # 초록색 박스 그리기
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 확신도(Score) 텍스트 쓰기
            text = f'Vehicle: {score:.2f}'
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    _, im_png = cv2.imencode('.png', img)
    

    # StreamingResponse를 사용하면 브라우저에 이미지를 바로 쏴줄 수 있다.
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type='image/png')