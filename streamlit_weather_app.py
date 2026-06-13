import streamlit as st
from google import genai

def call_gemini_api(question, api_key, city, weather):
    """Gemini API를 호출하여 날씨 관련 답변을 생성합니다."""
    try:
        client = genai.Client(api_key=api_key)
        model_id = "gemini-1.5-flash"
        prompt = f"""
        너는 {city}의 날씨 전문가 챗봇이야. 현재 {city}의 날씨 정보는 다음과 같아:
        - 온도: {weather['temperature']}°C
        - 상태: {weather['description']}
        - 습도: {weather['humidity']}% 
        - 풍속: {weather['wind_speed']} m/s
        이 정보를 바탕으로 사용자의 질문에 친절하고 전문적으로 답변해줘: {question}
        """
        response = client.models.generate_content(model=model_id, contents=prompt)
        return response.text
    except Exception as e:
        return f"API 호출 중 오류가 발생했습니다: {str(e)}"

def get_dummy_weather_data(city):
    """더미 날씨 데이터를 반환합니다."""
    if city == "경기도 성남시 분당구":
        return {
            "temperature": 25,
            "description": "맑음",
            "humidity": 60,
            "wind_speed": 5
        }
    else:
        return None

st.title("오늘의 날씨")

# OpenWeatherMap API 키를 위한 placeholder (나중에 실제 키로 대체)
# OPENWEATHER_API_KEY = "YOUR_API_KEY_HERE"

city_name = "경기도 성남시 분당구"

if city_name:
    st.write(f"## {city_name}의 날씨 정보")

    # 실제 API 연동 시 주석 해제하여 사용
    # import requests
    # if OPENWEATHER_API_KEY:
    #     try:
    #         response = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={OPENWEATHER_API_KEY}&units=metric")
    #         response.raise_for_status() # HTTP 오류 발생 시 예외 발생
    #         data = response.json()
    #         temperature = data["main"]["temp"]
    #         description = data["weather"][0]["description"]
    #         humidity = data["main"]["humidity"]
    #         wind_speed = data["wind"]["speed"]
    #         st.metric("온도", f"{temperature}°C")
    #         st.write(f"날씨: {description}")
    #         st.write(f"습도: {humidity}""%")
    #         st.write(f"풍속: {wind_speed} m/s")
    #     except requests.exceptions.RequestException as e:
    #         st.error(f"날씨 정보를 가져오는 중 오류가 발생했습니다: {e}")
    #         st.warning("더미 데이터를 사용합니다.")
    #         weather_data = get_dummy_weather_data(city_name)
    #         if weather_data:
    #             st.metric("온도", f"{weather_data["temperature"]}°C")
    #             st.write(f"날씨: {weather_data["description"]}")
    #             st.write(f"습도: {weather_data["humidity"]}""%")
    #             st.write(f"풍속: {weather_data["wind_speed"]}")
    #         else:
    #             st.error("더미 날씨 정보도 가져올 수 없습니다.")
    # else:
    #     st.warning("OpenWeatherMap API 키가 설정되지 않았습니다. 더미 데이터를 사용합니다.")

    weather_data = get_dummy_weather_data(city_name)

    if weather_data:
        st.metric("온도", f"{weather_data['temperature']}°C")
        st.write(f"날씨: {weather_data['description']}")
        st.write(f"습도: {weather_data['humidity']}%")
        st.write(f"풍속: {weather_data['wind_speed']} m/s")
    else:
        st.error("날씨 정보를 가져올 수 없습니다.")
else:
    st.warning("도시 이름을 입력해주세요.")

st.subheader("날씨 챗봇")

# Google Gemini API 키를 위한 placeholder
GOOGLE_API_KEY = st.sidebar.text_input("Gemini API 키를 입력하세요", type="password")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input("날씨에 대해 질문해주세요:")

if user_question:
    if not GOOGLE_API_KEY:
        st.warning("Gemini API 키가 설정되지 않았습니다. 챗봇이 작동하지 않습니다.")
    else:
        st.session_state.chat_history.append(("user", user_question))

        # 여기에 Gemini API 호출 로직 구현
        if weather_data:
            response = call_gemini_api(user_question, GOOGLE_API_KEY, city_name, weather_data)
            st.session_state.chat_history.append(("bot", response))
        else:
            st.error("현재 날씨 정보를 가져올 수 없어 챗봇이 답변하기 어렵습니다.")

for speaker, text in st.session_state.chat_history:
    if speaker == "user":
        st.write(f"**나:** {text}")
    else:
        st.write(f"**챗봇:** {text}")
