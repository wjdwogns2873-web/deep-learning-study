# Apple Silicon 맥미니(arm64)에 최적화된 경량 파이썬 이미지
FROM python:3.10-slim

# OpenCV 작동을 위한 Linux 필수 라이브러리 (libgl1-mesa-glx -> libgl1 으로 수정)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 및 모델 파일 복사
COPY . .

# FastAPI 실행 (8000 포트)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]