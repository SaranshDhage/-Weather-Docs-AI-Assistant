from typing import TypedDict, Literal, Optional, Dict, Any, List
from langgraph.graph import StateGraph, END
from router import route_query
from weather import fetch_weather, summarize_weather
from rag import retrieve_docs, synthesize_answer
from llm import get_llm, render_summary_prompt
from vectorstore import upsert_interaction


class AppState(TypedDict, total=False):
    query: str
    route: Literal["weather", "rag", "unknown"]
    city: Optional[str]
    weather_json: Optional[Dict[str, Any]]
    docs: Optional[List[Dict[str, Any]]]
    answer: Optional[str]
    meta: Dict[str, Any]


def router_node(state: AppState) -> AppState:
    route, city = route_query(state["query"])
    return {**state, "route": route, "city": city}


def weather_node(state: AppState) -> AppState:
    if state.get("route") == "weather" and not state.get("city"):
        raise ValueError(
            "Could not infer city from query. Try asking: 'What's the weather in <city>?'"
        )
    city = state.get("city", "")
    wjson = fetch_weather(city)
    summary = summarize_weather(state["query"], wjson)
    return {**state, "weather_json": wjson, "answer": summary}


def rag_node(state: AppState) -> AppState:
    docs = retrieve_docs(state["query"], k=4)
    answer = synthesize_answer(state["query"], docs)
    return {**state, "docs": docs, "answer": answer}


def finalize_node(state: AppState) -> AppState:
    llm = get_llm()
    prompt = render_summary_prompt(state.get("answer", ""))
    compact = (prompt | llm).invoke({})
    compact_text = getattr(compact, "content", str(compact))
    upsert_interaction(
        compact_text,
        metadata={"route": state.get("route"), "query": state.get("query")}
    )
    return state


def build_graph():
    g = StateGraph(AppState)

    g.add_node("router", router_node)
    g.add_node("weather", weather_node)
    g.add_node("rag", rag_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        lambda s: s.get("route", "unknown"),
        {
            "weather": "weather",
            "rag": "rag",
            "unknown": END
        }
    )

    g.add_edge("weather", "finalize")
    g.add_edge("rag", "finalize")
    g.add_edge("finalize", END)

    return g.compile()
