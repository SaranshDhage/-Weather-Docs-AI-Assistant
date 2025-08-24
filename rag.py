from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vectorstore import get_vectorstore
from llm import get_llm, render_rag_prompt


def retrieve_docs(question: str, k: int = 4) -> List[Document]:
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    return docs


def synthesize_answer(question: str, docs: List[Document]) -> str:
    # Merge document contents into a single context
    context_parts = [d.page_content[:1200] for d in docs] if docs else []
    context = "\n\n".join(
        context_parts) if context_parts else "(no relevant context)"

    # Get LLM and render a *clean* prompt
    llm = get_llm()
    prompt = render_rag_prompt(
        context=context,
        question=question
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})
