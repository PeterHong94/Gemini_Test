import argparse
import requests

API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

DEFAULT_CITY = "Seoul,KR"


def get_weather(city_name: str) -> dict:
    params = {
        "q": city_name,
        "appid": API_KEY,
        "units": "metric",
        "lang": "kr"
    }
    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def format_weather(data: dict) -> str:
    name = data.get("name")
    country = data.get("sys", {}).get("country")
    weather = data.get("weather", [{}])[0]
    main = data.get("main", {})
    wind = data.get("wind", {})

    lines = [
        f"현재 위치: {name}, {country}",
        f"날씨: {weather.get('description', '정보 없음')}",
        f"기온: {main.get('temp', 'N/A')}°C",
        f"체감 온도: {main.get('feels_like', 'N/A')}°C",
        f"최저 기온: {main.get('temp_min', 'N/A')}°C",
        f"최고 기온: {main.get('temp_max', 'N/A')}°C",
        f"습도: {main.get('humidity', 'N/A')}%",
        f"풍속: {wind.get('speed', 'N/A')} m/s",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="한국 현재 날씨를 보여주는 간단한 샘플 프로그램입니다.")
    parser.add_argument("city", nargs="?", default=DEFAULT_CITY, help="도시 이름 (예: Seoul,KR 또는 Busan,KR)")
    args = parser.parse_args()

    if API_KEY == "YOUR_OPENWEATHERMAP_API_KEY":
        print("OpenWeatherMap API 키를 'API_KEY' 변수에 설정한 다음 실행하세요.")
        return

    try:
        weather_data = get_weather(args.city)
        print(format_weather(weather_data))
    except requests.exceptions.RequestException as error:
        print(f"날씨 정보를 가져오는 중 오류가 발생했습니다: {error}")


if __name__ == "__main__":
    main()
