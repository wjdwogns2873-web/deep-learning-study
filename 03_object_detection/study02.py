from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


app = FastAPI()

# HTML 템플릿과 정적 파일 경로 세팅
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static") # 필요 시 사용

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # templates/index.html 페이지를 렌더링하여 반환
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_scene(
    request: Request, 
    file: UploadFile = File(...), 
    threshold: float = Form(0.5) # HTML Input Form에서 받아올 파라미터
):
    # threshold 값을 이용해 Object Detection 필터링 조건 제어 가능
    file_bytes = await file.read()

    result_message = f"Confidence Threshold **{threshold}** 기준으로 분석을 성공적으로 완료했습니다."

    response_data = {
        "filename": file.filename,
        "used_threshold": threshold,
        "message": f"Threshold {threshold} 기준으로 분석을 완료했습니다."
    }

    return templates.TemplateResponse(
        request=request, 
        name="result.html", 
        context={
            "filename": file.filename, 
            "threshold": threshold, 
            "message": result_message
        }
    )