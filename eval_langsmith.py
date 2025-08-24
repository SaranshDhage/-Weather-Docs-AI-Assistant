"""Minimal LangSmith evaluation demo.

- Requires LANGSMITH_API_KEY and tracing enabled.
- Compares model answers against simple expected references using LLM-as-judge.
"""
import os
from typing import Dict
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from llm import get_llm
from graph import build_graph

def task_runner(example: Dict) -> str:
    graph = build_graph()
    out = graph.invoke({"query": example["input"], "meta": {}})
    return out.get("answer", "")

def main():
    # Create a tiny inline dataset
    dataset = [
        {"input": "What's the weather in London right now?"},
        {"input": "Summarize the PDF's main contribution."},
    ]

    # LLM-as-judge (uses same Groq LLM behind the scenes)
    judge = LangChainStringEvaluator("criteria",
        config = {
            "criteria": {
                "helpfulness": "Is the answer helpful to the user?",
                "groundedness": "Is the answer grounded in the provided context or task?",
            },
            "llm": get_llm(temperature=0.0)
        }
    )

    results = evaluate(
        task=task_runner,
        data=dataset,
        evaluators=[judge],
        experiment_prefix="weather-rag-groq-demo",
    )
    print("Evaluation complete. See LangSmith for run details.")
    return results

if __name__ == "__main__":
    main()
