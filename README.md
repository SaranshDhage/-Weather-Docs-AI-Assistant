# Weather-Docs-AI-Assistant

A fully functional, modular AI pipeline demonstrating:

- **Agentic routing with LangGraph** (weather vs. RAG on a PDF)
- **Weather** via OpenWeatherMap API
- **RAG** with Qdrant vector DB and HuggingFace embeddings
- **LLM processing** via **Groq** (LangChain `ChatGroq`)
- **Embeddings** generation & storage in Qdrant (PDF chunks + interaction summaries)
- **LangSmith** tracing & evaluation
- **Clean, testable code** with `pytest`
- **Streamlit** UI chat demo

> ✅ This code is compatible with **Groq** (not OpenAI).

---

## 1) Project Structure

```
langgraph_rag_weather_groq/
├─ app.py                      # Streamlit UI
├─ graph.py                    # LangGraph build (nodes + state)
├─ router.py                   # LLM/heuristic router (weather vs rag)
├─ weather.py                  # OpenWeatherMap client + summarizer
├─ rag.py                      # PDF ingestion, retrieval, QA synthesis
├─ vectorstore.py              # Qdrant setup, upsert, retrieval helpers
├─ llm.py                      # Groq LLM wrapper and prompts
├─ embeddings.py               # HF embeddings
├─ settings.py                 # Env + configuration
├─ eval_langsmith.py           # Example LangSmith evaluation script
├─ requirements.txt
├─ .env.example
├─ tests/
│  ├─ test_router.py
│  ├─ test_weather.py
│  └─ test_rag.py
└─ data/
   └─ sample.pdf               # Optional sample placeholder
```

---

## 2) Setup

### Prerequisites

- Python 3.10+
- A running **Qdrant** server (local via Docker or Qdrant Cloud).
- API keys:
  - `GROQ_API_KEY`
  - `OPENWEATHER_API_KEY`
  - `LANGSMITH_API_KEY` (optional, for evaluation + tracing)

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

Create `.env` (copy from `.env.example`) and fill values:

```bash
cp .env.example .env
```

**.env keys**

- `GROQ_API_KEY=`
- `GROQ_MODEL=llama3-70b-8192`
- `OPENWEATHER_API_KEY=`
- `QDRANT_URL=`
- `QDRANT_API_KEY=`
- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_PROJECT=weather-rag-groq-demo`
- `LANGSMITH_API_KEY=`

---

## 3) Run the Streamlit UI

```bash
streamlit run app.py
```

- Upload a PDF (once) to ingest into Qdrant.
- Ask natural questions—router will decide between:
  - Weather (e.g., "What's the weather in Mumbai ?")
  - RAG over your PDF (e.g., "Summarize section 2" or "What does the paper say about ...?")

---

## 4) Tests

```bash
pytest -q
```

> Tests use **mocks** for external services so they run locally without API calls.

---

## 5) LangSmith Evaluation

- Make sure `LANGSMITH_API_KEY` is set and tracing turned on.
- Example evaluator run:

```bash
python eval_langsmith.py
```

This creates a lightweight evaluation that judges outputs for helpfulness/groundedness using an LLM-as-judge approach.

---

## 6) Notes & Decisions

- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace for simplicity & low cost. You can swap the model in `embeddings.py`.
- **Collections**:
  - `docs`: PDF chunks
  - `interactions`: Summaries of each final response (weather or RAG) so the app can RAG over its own past outputs.
- **Routing**: Uses Groq LLM classification with regex fallback for reliability.
- **LLM**: Defaults to `llama3-70b-8192`. Adjust via env.
- **Security**: Keys are read from env; never hardcode credentials.
- **Clean code**: Modules are small, typed, and unit-tested via mocks.

---

## 7) Deliverables Recap

- ✅ Python code (this repo)
- ✅ README (this file)
- ✅ LangSmith logs 
- ✅ Tests (`pytest`)
- ✅ Streamlit UI (`app.py`)
- ✅ Evaluation script (`eval_langsmith.py`)
- ✅ You can record Loom explaining evaluation & code after running locally

---

## 8) Troubleshooting

- **PDF ingestion**: First time will download the HF embedding model; ensure internet access.
- **Qdrant auth**: Cloud requires API key; local dev does not.
- **Groq model**: If the default model string is not available in your account, pick another (e.g., `llama-3.1-70b`).
