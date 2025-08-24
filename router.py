from typing import Literal, Tuple
import re
from llm import get_llm
from langchain_core.prompts import ChatPromptTemplate

Route = Literal["weather","rag","unknown"]

ROUTE_PROMPT = ChatPromptTemplate.from_template(
    "Classify the user query into one of: 'weather' or 'rag'.\n"
    "If about current weather, forecasts, temperature, rain, humidity, or a city, choose 'weather'.\n"
    "If about a PDF/document content, knowledge, or asking questions unrelated to weather, choose 'rag'.\n"
    "Return only the single word label.\n\n"
    "Query: {query}"
)

CITY_PATTERN = re.compile(r"(?:weather|temperature|climate|forecast)\s*(?:in|at|for)?\s*([A-Za-z\s]+)", re.I)

def heuristic_city(query: str) -> str | None:
    m = CITY_PATTERN.search(query)
    if m:
        city = m.group(1).strip()
        # Strip trailing punctuation
        city = re.sub(r"[\?\.!]+$", "", city).strip()
        # Avoid generic words
        if len(city) >= 2:
            return city
    return None

def route_query(query: str) -> Tuple[Route, str | None]:
    # First, cheap heuristic: any overt weather keywords?
    if re.search(r"\b(weather|temperature|forecast|humidity|rain|wind)\b", query, re.I):
        return "weather", heuristic_city(query)

    # Otherwise ask the LLM (Groq) to route
    llm = get_llm(temperature=0)
    try:
        out = (ROUTE_PROMPT | llm).invoke({"query": query})
        label = out.content.strip().lower()
        if label not in {"weather","rag"}:
            raise ValueError("invalid label")
        if label == "weather":
            return "weather", heuristic_city(query)
        return "rag", None
    except Exception:
        # Fallback heuristic
        return ("rag", None)
