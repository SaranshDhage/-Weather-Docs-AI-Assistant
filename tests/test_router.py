from router import route_query, heuristic_city
from router import get_llm


def test_router_heuristic_weather():
    route, city = route_query("What is the weather in Delhi?")
    assert route == "weather"
    assert city and "delhi" in city.lower()


def test_router_llm_fallback_rag(monkeypatch):
    # Force fallback by raising exception in LLM call
    def boom(*args, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr("router.get_llm", lambda *a, **k: boom)

    route, city = route_query(
        "Explain section 2 of the document about retrieval.")
    assert route == "rag"
    assert city is None


def test_heuristic_city():
    assert heuristic_city("weather in Pune") == "Pune"
    assert heuristic_city("temperature at London?") == "London"
    assert heuristic_city("tell me weather") is None
