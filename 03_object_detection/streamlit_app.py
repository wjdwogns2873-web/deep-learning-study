import streamlit as st
import requests
from PIL import Image
import io

# 웹 페이지 제목 및 설명 설정
st.set_page_config(page_title="과일 신선도 탐지 시스템", layout="centered")
st.title("과일 신선도 실시간 판별 대시보드")
st.write("로지스틱스 현장용 경량화 YOLOv8 모델 구동 중")

st.divider() # 화면 구분선

# 이미지 업로드 컴포넌트 생성
uploaded_file = st.file_uploader("판별할 과일 사진을 업로드하세요!", type=["jpg", "jpeg", "png", "webp"])

# FastAPI 서버 주소
FastAPI_URL = "http://localhost:8000/predict"
# FastAPI_URL = "https://annex-heaviness-five.ngrok-free.dev/predict"


if uploaded_file is not None:
    # 업로드한 원본 이미지를 먼저 화면에 표시
    col1, col2 = st.columns(2)

    with col1: 
        st.subheader("원본 이미지")
        orig_image = Image.open(uploaded_file)
        st.image(orig_image, use_container_width=True)

    # FastAPI 서버로 이미지 전송
    img_byte_arr = io.BytesIO()
    orig_image.save(img_byte_arr, format=orig_image.format if orig_image.format else "JPEG")

    files = {"file": (uploaded_file.name, img_byte_arr.getvalue(), uploaded_file.type)}

    with st.spinner("FastAPI 서버에서 AI 감지 및 좌표 그리기 작업 중"):
        try:
            response = requests.post(FastAPI_URL, files=files)

            if response.status_code == 200:
                # 서버가 전송해 준 바운딩 박스가 그려진 이미지 바이트 받기
                result_bytes = response.content
                result_image = Image.open(io.BytesIO(result_bytes))

                # 나란히 결과 이미지 배치
                with col2: 
                    st.subheader("탐지 결과 (YOLOv8)")
                    st.image(result_image, use_container_width=True)

                st.success("탐지 및 결과 시각화가 완료되었습니다")
            else:
                st.error("서버 응답 실패. FastAPI 코드를 확인해 주세요")
        except requests.exceptions.ConnectionError:
            st.error("FastAPI 서버가 켜져 있지 않습니다. uvicorn 서버를 먼저 켜주세요")