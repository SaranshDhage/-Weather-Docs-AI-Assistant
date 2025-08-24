from typing import Any, Dict, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from settings import settings


def get_llm(temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(
        temperature=temperature,
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        # Optional: request batching etc. could be added here
    )


def render_weather_prompt(data: Dict[str, Any], user_query: str) -> ChatPromptTemplate:
    template = (
        """You are a helpful assistant. Summarize the current weather clearly and concisely.
"""
        """User asked: {user_query}
"""
        """Weather JSON:
{weather_json}
"""
        """Return a user-friendly, actionable summary in 3-6 sentences, with °C and any alerts."""
    )
    return ChatPromptTemplate.from_template(template).partial(
        weather_json=str(data), user_query=user_query
    )


def render_rag_prompt(context: str, question: str) -> ChatPromptTemplate:
    template = (
        """You are an assistant.
Go through the document to see what is given, understand and answer the question accordingly.

Question: {question}

Context:
{context}

Answer in plain language without mentioning chunks, documents, or sources. 
Be clear, concise, and natural."""
    )
    return ChatPromptTemplate.from_template(template).partial(
        context=context, question=question
    )


def render_summary_prompt(answer: str) -> ChatPromptTemplate:
    template = (
        """Summarize the following response into a compact 1-2 sentence nugget for future retrieval:

{answer}"""
    )
    return ChatPromptTemplate.from_template(template).partial(answer=answer)
