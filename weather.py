from typing import Dict, Any
import requests
import re

from settings import settings
from llm import get_llm, render_weather_prompt

OWM_URL = "https://api.openweathermap.org/data/2.5/weather"


def clean_city_name(query: str) -> str:
    """
    Extracts the most likely city name from a user query.
    Removes filler words like 'now', 'today', 'current', 'weather', 'in'.
    """
    query = query.lower().strip()

    # Remove common filler words
    query = re.sub(r"\b(now|today|current|weather|in)\b", "", query)

    # Collapse multiple spaces
    return re.sub(r"\s+", " ", query).strip()


def fetch_weather(city: str, units: str = "metric", lang: str = "en") -> Dict[str, Any]:
    """
    Calls OpenWeatherMap API and returns the raw weather JSON.
    """
    clean_city = clean_city_name(city)
    params = {
        "q": clean_city,
        "appid": settings.openweather_api_key,
        "units": units,
        "lang": lang,
    }

    resp = requests.get(OWM_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def summarize_weather(user_query: str, weather_json: Dict[str, Any]) -> str:
    """
    Summarizes weather data into a natural language response using LLM.
    """
    llm = get_llm()
    prompt = render_weather_prompt(weather_json, user_query)
    chain = prompt | llm
    response = chain.invoke({})

    # Handle both LangChain and raw string return types
    if hasattr(response, "content"):
        return response.content
    return str(response)
