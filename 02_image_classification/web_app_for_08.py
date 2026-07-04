import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import requests
from PIL import Image
import io
import torch

st.set_page_config(page_title="🍎 🍌 🍊 분류기", page_icon="🍎 🍌 🍊", layout="centered")
st.title("🍎 🍌 🍊 과일 신선도 판별 시스템")
st.write("🍎 🍌 🍊 이미지를 업로드하면 AI 모델이 신선도를 분석합니다.")
st.markdown("---")

FASTAPI_URL = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("이미지를 업로드하세요. jpg, jpeg, png", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("업로드된 이미지")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("AI 분석 결과")

        if st.button("신선도 분류하기", type="primary"):
            with st.spinner("FastAPI 서버와 통신중.."):
                try:
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                    img_byte_arr = img_byte_arr.getvalue()

                    files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}

                    response = requests.post(FASTAPI_URL, files=files)

                    if response.status_code == 200:
                        result = response.json()
                        prediction = result.get("prediction", "알 수 없음")
                        confidence = float(result.get("confidence", "알 수 없음"))
                        second_predicton = result.get("second_predicton", "알 수 없음")
                        second_prob = float(result.get("second_prob", "알 수 없음"))

                        classes = ["fresh_apple", "rotten_apple", "fresh_banana", 
                                   "rotten_banana", "fresh_orange", "rotten_orange"]
                        # classes_map = {name: i for i, name in enumerate(classes)}
                        # i_to_name_map = {i: name for i, name in enumerate(classes_map)}
                        
                        # print(f"****confidence type: {type(confidence)}, second_prob type: {type(second_prob)}")
                        # print(f"****confidence: {confidence}, second_prob: {second_prob}")
                        comment = ""
                        
                        if confidence < 85:
                            comment = f"{confidence}% 확률로 {prediction}이지만..\n{second_prob}% 확률로   {second_predicton}"
                        else:
                            comment = f"{confidence}%확률로 {prediction}"

                        if prediction in classes:
                            st.success(comment)
                            st.balloons()
                        else:
                            st.warning(f"예측값 확인 필요: {prediction}")

                    else:
                        st.warning(f"Server Error. status code: {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("FastAPI 서버와 통신할 수 없습니다.")