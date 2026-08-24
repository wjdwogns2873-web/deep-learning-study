import cv2
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

import time
import tempfile

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = YOLO("runs/fruits_1epoch/weights/best.pt")

UPLOAD_DIR = Path("temp_videos")
UPLOAD_DIR.mkdir(exist_ok=True)

# 각 클라이언트의 스트리밍 활성화 상태를 관리하는 딕셔너리
active_streams = {}

def generate_frames(video_bytes: bytes, session_id: str):
    # NamedTemporaryFile을 사용하면 하드디스크에 물리 파일을 쓰지 않고 메모리 버퍼 영역에서 객체를 시뮬레이션함.
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        temp_video.seek(0)

        cap = cv2.VideoCapture(temp_video.name)

        # FPS 계산을 위한 이전 시간 기록 변수
        prev_time = 0

        while cap.isOpened():
            # 사용자가 프론트엔드에서 종료를 누르면 루프를 즉시 탈출
            if not active_streams.get(session_id, True): break

            success, frame = cap.read()
            if not success: break

            # YOLOv8 추론
            results = model(frame, stream=True)
            annotated_frame = frame

            # 실시간 객체 수 카운팅 변수 초기화
            object_counts = {}

            for r in results:
                annotated_frame = r.plot()

                # 내부 boxes 데이터 구조에서 각 물체의 클래스 ID(cls) 목록을 가져온다
                if hasattr(r, "boxes") and r.boxes is not None:
                    classes = r.boxes.cls.int().tolist()
                    for class_id in classes:
                        class_name = model.names[class_id] # 예: 'person', 'car'
                        object_counts[class_name] = object_counts.get(class_name, 0) + 1

            # 실시간 FPS 계산 (현재 시간 - 이전 프레임 처리 시간)
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time

            # 프레임 좌측 상단에 텍스트 데이터 합성 (OpenCV text 오버레이)
            info_text = f"FPS: {fps:.1f} | " + ", ".join([f"{k}: {v}" for k, v in object_counts.items()])
            cv2.putText(
                annotated_frame, info_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
            )

            # JPEG 변환 및 바이트 전송
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret: continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()

    # 루프가 정상 종료되거나 강제 중지되었을 때 세션 정보 정리
    if session_id in active_streams:
        del active_streams[session_id]
            
            
# def generate_frames(video_path: str):
#     # 비디오에서 프레임을 읽어 YOLO로 추론 후 byte 스트림으로 변환합니다
#     cap = cv2.VideoCapture(video_path)

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success: break

#         # YOLOv8 추론 수행 (render/plot 결과를 가져옴)
#         # persist=True 옵션은 추적 기능을 활성화
#         results = model(frame, stream=True)

#         for r in results:
#             # 네모박스가 그려진 프레임을 가져옴
#             annotated_frame = r.plot()

#             # 프레임을 JPEG 포맷으로 인코딩
#             ret, buffer = cv2.imencode('.jpg', annotated_frame)
#             if not ret: continue
    
#             frame_bytes = buffer.tobytes()
    
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#     cap.release()

# 처음 접속시 업로드 페이지 렌더링
@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html"
    )

# @app.post("/upload")
# async def upload_video(request: Request, file: UploadFile = File(...)):
#     # 파일을 디스크에 저장하지 않고 바이너리 전체를 한번에 메모리로 읽어들입니다.
#     video_bytes = await file.read()

#     # 세션을 식별할 고유 키 생성
#     session_id = str(time.time()).replace(".", "")
#     active_streams[session_id] = True # 스트림 가동 상태 신호 ON

#     return templates.TemplateResponse(
#         request=request, 
#         name="stream.html", 
#         context={"session_id": session_id}
#     )

# 클라이언트가 파일 제출시 유니크 파일로 저장 후, 스트리밍 페이지로 이동
# @app.post("/upload")
# async def upload_video(request: Request, file: UploadFile = File(...)):
#     file_extension = Path(file.filename).suffix # .mp4
#     unique_filename = f"{uuid.uuid4()}{file_extension}"
#     temp_file_path = UPLOAD_DIR / unique_filename

#     with open(temp_file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # 스트리밍 화면(stream.html)을 보여주면서, 비디오 파일명을 변수로 넘겨줌
#     return templates.TemplateResponse(
#         request=request, 
#         name="stream.html", 
#         context={"video_id": unique_filename}
#     )

@app.get("/video_feed/{session_id}")
async def video_feed(session_id: str, request: Request):
    # 전역 딕셔너리에서 세션을 확인하고 프레임 제너레이터를 호출합니다.
    return StreamingResponse(
        generate_frames(request.app.state.last_bytes if hasattr(request.app.state, 'last_bytes') else b'', session_id), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )



# 스트리밍 영상 전용 엔드포인트
# @app.get("/video_feed/{video_id}")
# async def video_feed(video_id: str):
#     video_path = UPLOAD_DIR / video_id
#     return StreamingResponse(
#         generate_frames(str(video_path)), 
#         media_type="multipart/x-mixed-replace; boundary=frame"
#     )


# [비동기 엽로드 전용 데이터 보관을 위한 임시 장치]
@app.post("/upload_memory/{session_id}")
async def upload_memory(session_id: str, request: Request, file: UploadFile = File(...)):
    # 최적화 처리를 위해 업로드된 바이너리를 일시적으로 상태 구조에 적재합니다.
    request.app.state.last_bytes = await file.read()
    active_streams[session_id] = True
    return {"status": "ready"}

# 사용자가 웹화면에서 중지 버튼을 누르면 서버의 플래그를 꺼버리는 API
@app.post("/stop_stream/{session_id}")
async def stop_stream(session_id: str):
    if session_id in active_streams:
        active_streams[session_id] = False # False를 주어 루프가 종료되도록 유도
    return {"status": "stopped"}


@app.get("/video_feed_page")
async def video_feed_page(request: Request, session_id: str):
    return templates.TemplateResponse(
        request=request, 
        name="stream.html", 
        context={"session_id": session_id}
    )

# @app.post("/video_feed")
# async def video_feed(file: UploadFile = File(...)):

#     file_extension = Path(file.filename).suffix

#     # UUID를 활용한 유니크한 파일명 생성
#     unique_filename = f"{uuid.uuid4()}{file_extension}"
    
#     # 클라이언트가 업로드한 비디오를 스트리밍 응답으로 반환합니다.
#     # 1. 업로드된 파일을 디스크에 임시 저장
#     temp_file_path = UPLOAD_DIR / unique_filename
#     with open(temp_file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # 2. StreamingResponse를 통해 실시간 프레임 전달
#     return StreamingResponse(
#         generate_frames(str(temp_file_path)), 
#         media_type="multipart/x-mixed-replace; boundary=frame"
#     )

