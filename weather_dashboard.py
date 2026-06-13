import os
from typing import Any

import requests
import streamlit as st


BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
ICON_URL = "https://openweathermap.org/img/wn/{icon}@2x.png"
KMA_JSON_URL = "https://apihub.kma.go.kr/api/json"

KOREAN_CITIES = {
    "서울": "Seoul,KR",
    "부산": "Busan,KR",
    "인천": "Incheon,KR",
    "대구": "Daegu,KR",
    "대전": "Daejeon,KR",
    "광주": "Gwangju,KR",
    "울산": "Ulsan,KR",
    "세종": "Sejong,KR",
    "수원": "Suwon,KR",
    "성남": "Seongnam,KR",
    "고양": "Goyang,KR",
    "용인": "Yongin,KR",
    "청주": "Cheongju,KR",
    "전주": "Jeonju,KR",
    "천안": "Cheonan,KR",
    "포항": "Pohang,KR",
    "창원": "Changwon,KR",
    "제주": "Jeju,KR",
}


class WeatherApiError(Exception):
    """OpenWeatherMap API 오류를 화면에 알기 쉽게 보여주기 위한 예외입니다."""


@st.cache_data(ttl=300, show_spinner=False)
def get_current_weather(city_query: str, api_key: str) -> dict[str, Any]:
    """OpenWeatherMap에서 현재 날씨를 가져옵니다. 5분 동안 결과를 캐시합니다."""
    params = {
        "q": city_query,
        "appid": api_key,
        "units": "metric",
        "lang": "kr",
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        data = response.json()
    except requests.exceptions.Timeout as exc:
        raise WeatherApiError("요청 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.") from exc
    except requests.exceptions.RequestException as exc:
        raise WeatherApiError("날씨 정보를 가져오는 중 네트워크 오류가 발생했습니다.") from exc
    except ValueError as exc:
        raise WeatherApiError("날씨 서버의 응답을 읽을 수 없습니다.") from exc

    if response.status_code != 200:
        message = data.get("message", "알 수 없는 오류")
        raise WeatherApiError(f"날씨 정보를 가져오지 못했습니다: {message}")

    return data

@st.cache_data(ttl=300, show_spinner=False)
def get_kma_weather(api_key: str, api_url: str = KMA_JSON_URL) -> dict[str, Any]:
    """기상청 JSON API를 호출해 응답을 JSON으로 반환합니다."""
    params = {
        "authKey": api_key,
    }

    try:
        response = requests.get(api_url, params=params, timeout=10)
        data = response.json()
    except requests.exceptions.Timeout as exc:
        raise WeatherApiError("요청 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.") from exc
    except requests.exceptions.RequestException as exc:
        raise WeatherApiError("날씨 정보를 가져오는 중 네트워크 오류가 발생했습니다.") from exc
    except ValueError as exc:
        raise WeatherApiError("기상청 응답을 JSON으로 변환할 수 없습니다.") from exc

    if response.status_code != 200:
        message = data.get("message", "알 수 없는 오류")
        raise WeatherApiError(f"기상청 데이터를 가져오지 못했습니다: {message}")

    return data


def format_temp(value: float | int | None) -> str:
    """온도를 소수점 한 자리까지 보기 좋게 표시합니다."""
    if value is None:
        return "N/A"
    return f"{value:.1f}°C"


def main() -> None:
    st.set_page_config(page_title="한국 날씨 대시보드", page_icon="☀️", layout="wide")

    env_api_key = os.getenv("OPENWEATHERMAP_API_KEY", "")
    env_kma_key = os.getenv("KMA_AUTH_KEY", "")

    st.sidebar.header("설정")
    api_provider = st.sidebar.selectbox(
        "API 제공자 선택",
        ["OpenWeatherMap", "기상청 JSON"],
        help="OpenWeatherMap 또는 기상청 JSON API를 선택하세요.",
    )

    if api_provider == "OpenWeatherMap":
        api_key = st.sidebar.text_input(
            "OpenWeatherMap API 키",
            value=env_api_key,
            type="password",
            help="환경 변수 OPENWEATHERMAP_API_KEY를 설정하면 자동으로 불러옵니다.",
        )
    else:
        api_key = st.sidebar.text_input(
            "기상청 authKey",
            value=env_kma_key,
            type="password",
            help="기상청 API authKey를 입력하세요.",
        )
        kma_url = st.sidebar.text_input(
            "기상청 JSON API URL",
            value=KMA_JSON_URL,
            help="기상청 JSON API 엔드포인트를 입력합니다.",
        )

    selected_city = st.sidebar.selectbox("도시 선택", list(KOREAN_CITIES.keys()))
    custom_city = st.sidebar.text_input(
        "직접 입력",
        placeholder="예: Gangneung,KR 또는 강릉",
        help="입력하면 드롭다운 선택보다 우선합니다.",
    )

    city_query = custom_city.strip() or KOREAN_CITIES[selected_city]

    st.title("한국 현재 날씨 대시보드")
    st.caption("JSON API로 날씨 데이터를 읽어오는 샘플입니다.")

    if not api_key:
        st.info("사이드바에 API 키를 입력하거나 환경 변수를 설정해 주세요.")
        return

    with st.spinner("날씨 정보를 불러오는 중입니다..."):
        try:
            if api_provider == "OpenWeatherMap":
                data = get_current_weather(city_query, api_key)
            else:
                data = get_kma_weather(api_key, kma_url)
        except WeatherApiError as exc:
            st.error(str(exc))
            st.warning("API 키나 요청 URL을 확인해 주세요.")
            return

    if api_provider == "OpenWeatherMap":
        weather = data.get("weather", [{}])[0]
        main_weather = data.get("main", {})
        wind = data.get("wind", {})
        sys = data.get("sys", {})

        city_name = data.get("name", city_query)
        country = sys.get("country", "N/A")
        description = weather.get("description", "정보 없음")
        icon = weather.get("icon")

        header_col, icon_col = st.columns([4, 1])
        with header_col:
            st.subheader(f"{city_name}, {country}")
            st.write(f"현재 날씨: **{description}**")
        with icon_col:
            if icon:
                st.image(ICON_URL.format(icon=icon), width=100)

        temp_col, feels_col, min_col, max_col = st.columns(4)
        temp_col.metric("현재 기온", format_temp(main_weather.get("temp")))
        feels_col.metric("체감 온도", format_temp(main_weather.get("feels_like")))
        min_col.metric("최저 기온", format_temp(main_weather.get("temp_min")))
        max_col.metric("최고 기온", format_temp(main_weather.get("temp_max")))

        humidity_col, wind_col = st.columns(2)
        humidity_col.metric("습도", f"{main_weather.get('humidity', 'N/A')}%")
        wind_col.metric("풍속", f"{wind.get('speed', 'N/A')} m/s")
    else:
        st.warning("기상청 JSON API는 응답 구조가 다양하므로 원본 JSON을 확인하여 필요한 값을 추가로 파싱하세요.")
        st.write("### 기상청 JSON 원본 응답")

    with st.expander("원본 응답 보기"):
        st.json(data)


if __name__ == "__main__":
    main()
