from google import genai
import streamlit as st

# Streamlit 앱에서 API 키를 입력받는다고 가정
# GOOGLE_API_KEY = st.sidebar.text_input("Gemini API 키를 입력하세요", type="password")
# 실제 테스트 시에는 여기에 직접 API 키를 넣거나, 환경 변수에서 가져오세요.
GOOGLE_API_KEY = "YOUR_ACTUAL_API_KEY" # 여기에 실제 API 키를 입력하세요.

if GOOGLE_API_KEY:
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        print("사용 가능한 Gemini 모델 목록:")
        for m in client.models.list():
            print(f"- {m.name}")
    except Exception as e:
        print(f"모델 목록을 가져오는 중 오류 발생: {e}")
else:
    print("Gemini API 키를 설정해주세요.")
