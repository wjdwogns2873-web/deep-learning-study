import streamlit as st
import requests
from PIL import Image
import io

# 1. 웹페이지 상단 레이아웃 설정
st.set_page_config(page_title="사과 신선도 분류기", page_icon="🍎", layout="centered")

st.title("🍎 과일 신선도 판별 시스템")
st.write("사과 이미지를 업로드하면 AI 모델이 신선도를 분석합니다.")
st.markdown("---")

# FastAPI 서버 주소 (로컬에서 실행 중인 주소)
# 만약 ngrok 주소를 쓰고 계시다면 "https://xxxx.ngrok-free.app/predict" 형태로 변경 가능합니다.
FASTAPI_URL = "http://127.0.0.1:8000/predict"

# 2. 이미지 업로드 UI 생성
uploaded_file = st.file_uploader("사과 이미지를 업로드하세요 (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 사용자가 업로드한 이미지를 화면에 표시하기 위해 PIL 이미지로 변환
    image = Image.open(uploaded_file)
    
    # 두 개의 열(Column)을 만들어 좌측에는 이미지, 우측에는 결과창을 배치
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 업로드된 이미지")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("🤖 AI 분석 결과")
        
        # 사용자가 "분석 시작" 버튼을 누르면 작동
        if st.button("신선도 분석하기", type="primary"):
            with st.spinner("FastAPI 서버와 통신 중..."):
                try:
                    # 3. Streamlit에 로드된 파일의 바이트 데이터를 추출
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format if image.format else "JPEG")
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    # 4. FastAPI의 주석에 맞춰 {'file': (파일명, 바이트, 원래 확장자)} 형태로 전송 준비
                    files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}
                    
                    # 5. FastAPI 서버로 POST 요청 보내기
                    response = requests.post(FASTAPI_URL, files=files)
                    
                    if response.status_code == 200:
                        # 결과 JSON 파싱 (예: {"prediction": "fresh"})
                        result = response.json()
                        prediction = result.get("prediction", "알 수 없음")
                        
                        # 6. 결과에 따른 예쁜 UI 출력
                        if prediction == "fresh":
                            st.success("✨ 분석 결과: **신선함 (Fresh)**")
                            st.balloons() # 축하 풍선 이펙트
                        elif prediction == "rotten":
                            st.error("🚨 분석 결과: **상함 (Rotten)**")
                        else:
                            st.warning(f"⚠️ 예측값 확인 필요: {prediction}")
                    else:
                        st.error(f"서버 에러 발생 (Status Code: {response.status_code})")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ FastAPI 서버에 연결할 수 없습니다. app.py가 실행 중인지 확인하세요.")