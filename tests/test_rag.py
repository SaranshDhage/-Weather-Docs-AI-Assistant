from rag import retrieve_docs, synthesize_answer
from langchain_core.documents import Document


def test_synthesize_answer_with_docs(monkeypatch):
    # Mock LLM as callable
    class DummyLLM:
        def __or__(self, other): return self
        def __call__(self, _): return "Answer with [chunk-1]"

    monkeypatch.setattr("rag.get_llm", lambda: DummyLLM())

    docs = [Document(page_content="This is content about AI.")]
    ans = synthesize_answer("What is AI?", docs)
    assert "Answer" in ans


def test_retrieve_docs_calls_vectorstore(monkeypatch):
    class DummyVS:
        def as_retriever(self, search_kwargs=None):
            class R:
                def invoke(self, q):
                    return [Document(page_content=f"Chunk for: {q}")]
            return R()
    monkeypatch.setattr("rag.get_vectorstore", lambda: DummyVS())

    docs = retrieve_docs("question", k=2)
    assert len(docs) == 1
    assert "Chunk for" in docs[0].page_content
