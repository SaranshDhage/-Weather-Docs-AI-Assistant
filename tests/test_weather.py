import types
from weather import fetch_weather, summarize_weather


def test_fetch_weather_builds_params(monkeypatch):
    calls = {}

    def fake_get(url, params=None, timeout=None):
        calls['url'] = url
        calls['params'] = params

        class R:
            def raise_for_status(self): pass
            def json(self): return {"main": {"temp": 30},
                                    "name": params.get('q', '?')}
        return R()
    monkeypatch.setattr("weather.requests.get", fake_get)

    data = fetch_weather("Pune")
    assert calls['params']['q'].lower() == "pune"
    assert 'appid' in calls['params']
    assert data['main']['temp'] == 30


def test_summarize_weather_uses_llm(monkeypatch):
    class Dummy:
        def __call__(self, _):
            return types.SimpleNamespace(content="It's hot in Pune at ~30°C.")

    monkeypatch.setattr("weather.get_llm", lambda: Dummy())

    summary = summarize_weather("weather in Pune", {"main": {"temp": 30}})
    assert "30" in summary
