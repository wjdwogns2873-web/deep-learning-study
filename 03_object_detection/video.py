import cv2
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import shutil
import os
from fastapi.staticfiles import StaticFiles

model = YOLO("runs/fruits_1epoch/weights/best.pt")

dir_path = "videos"
os.makedirs(dir_path, exist_ok=True)

app = FastAPI()

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # 동영상 파일인지 확장자 확인
    if not file.content_type.startswith("video/"):
        return {"error": "동영상 파일만 업로드할 수 있습니다."}

    # 저장할 경로 설정
    file_path = f"{dir_path}/{file.filename}"

    # 비동기로 파일 저장
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        # shutil.copyfileobj(file.file, buffer)

    return {"message": "업로드 완료", "saved_path": file_path}
    # return StreamingResponse(generate_frames(file_path), media_type="multipart/x-mixed-replace; boundary=frame")

def generate_frames(video_path: str):
    # 내 프로젝트 폴더에 있는 동영상 파일을 OpenCV로 엽니다.
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break # 영상이 끝나면 탈출

        # 1. YOLO에게 이 프레임 한 장을 추적하라고 던집니다
        results = model.track(frame, stream=True)

        for result in results:
            # 2. YOLO가 박스와 ID를 그린 결과 이미지(넘파이 배열)를 끄집어냅니다.
            annotated_frame = result.plot()

            # 넘파이 배열 상태의 이미지는 용량이 너무 크고 웹 브라우저가 직접 읽지 못한다.
            # 우리가 흔히 보는 .jpg포맷으로 이미지 용량을 압축하는 과정이다. 
            # 압축된 이진 데이터가 buffer에 담긴다.

            # 3. 이 이미지를 웹브라우저가 읽을 수 있게 JEPG 포맷으로 압축한다.
            # cv2.imencode는 성공 여부(ret)와 압축된 데이터(buffer)를 뱉습니다.
            ret, buffer = cv2.imencode(".jpg", annotated_frame)

            # 압축된 데이터 조각을 네트워크 파이프라인을 타고 전송할 수 있도록 순수한 바이트 알맹이로 최종 변환한다.
            
            # 4. 압축된 데이터를 바이트 형태로 변환합니다.
            frame_bytes = buffer.tobytes()

            # 5. FastAPI 스트리밍 규격에 맞게 껍데기를 씌워 무한리필(yield)합니다.
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()
    


@app.get("/video_feed")
def video_feed():
    # StreamingResponse에게 "1번 조각 함수를 실행해서 계속 뿜어져 나오는 데이터를 브라우저에 쏴줘"라고 명령합니다.
    # media_type은 브라우저에게 "이건 계속 바뀌는 프레임 조각들이야"라고 알려주는 고정 규칙입니다.
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

from fastapi.responses import HTMLResponse
@app.get("/")
def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>컨베이어 벨트 모니터링 시스템</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background-color: #f4f4f4; padding: 20px; }
            .container { background-color: white; padding: 30px; border-radius: 10px; display: inline-block; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            img { border: 4px solid #333; border-radius: 5px; max-width: 100%; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI 기반 컨베이어 벨트 실시간 과일 추적</h1>
            <p>YOLOv8 가중치 기반 객체 탐지 및 ID 추적 스트리밍 화면입니다.</p>
            <img src="/video_feed" width="800">
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)