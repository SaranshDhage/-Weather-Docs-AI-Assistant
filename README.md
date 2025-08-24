# Weather Docs AI Assistant

This project is an **AI-powered assistant** built with **LangChain**, **Groq LLMs**, **Qdrant vector database**, and a **Weather API**.  
It combines **document-based question answering (RAG pipeline)** with **real-time weather updates**, making it both a knowledge retrieval assistant and a utility tool.

---

## 🚀 Features
- **RAG Pipeline**: Retrieve answers from custom documents using embeddings and a vector database (Qdrant).
- **Weather Integration**: Fetch live weather updates using a Weather API.
- **LLM Integration**: Powered by Groq through LangChain for efficient reasoning.
- **Evaluation**: Integrated with LangSmith for tracing and debugging.
- **Modular Codebase**: Organized into independent components for LLMs, embeddings, vector stores, weather, and routing.

---

## 📂 Project Structure

```
LANGCHAIN_PROJECT/
│── data/                      # Store input documents for embeddings
│── tests/                     # Unit tests for each module
│   ├── test_rag.py             # Tests RAG pipeline
│   ├── test_router.py          # Tests routing logic
│   ├── test_weather.py         # Tests weather integration
│
│── .env                        # Environment variables (API keys etc.)
│── .gitignore                  # Ignore cache, venv, and secrets
│── app.py                      # Entry point (Streamlit app or CLI)
│── embeddings.py               # Handles document embeddings
│── eval_langsmith.py           # Evaluation & tracing with LangSmith
│── graph.py                    # Manages computation graphs / flow
│── llm.py                      # Loads and configures Groq LLM
│── rag.py                      # Core Retrieval-Augmented Generation pipeline
│── router.py                   # Directs queries to RAG or Weather
│── settings.py                 # Global configuration management
│── vectorstore.py              # Handles Qdrant vector DB operations
│── weather.py                  # Weather API integration
│── requirements.txt            # Python dependencies
│── README.md                   # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root and add:
```
GROQ_API_KEY=your_groq_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url
WEATHER_API_KEY=your_weather_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
```

### 5. Run the Application
If using Streamlit frontend:
```bash
streamlit run app.py
```

If running CLI script:
```bash
python app.py
```

---

## 🔄 How the System Works

1. **Document Ingestion**
   - Documents inside `data/` are embedded using the embeddings model.
   - Embeddings are stored inside **Qdrant vector database** (`vectorstore.py`).

2. **Query Routing**
   - User query is passed to the `router.py`.
   - If query is about **weather**, it is directed to `weather.py`.
   - Otherwise, query goes into the **RAG pipeline**.

3. **RAG Pipeline**
   - Query embeddings are generated (`embeddings.py`).
   - Relevant context chunks are retrieved from **Qdrant** (`vectorstore.py`).
   - Retrieved chunks are combined with the query to form a prompt (`rag.py`).
   - Groq LLM (`llm.py`) generates a final response.

4. **Weather API**
   - For weather queries, the request goes directly to the Weather API (`weather.py`).
   - Real-time weather details are returned to the user.

5. **LangSmith Evaluation**
   - Every request/response is logged (`eval_langsmith.py`).
   - Useful for debugging, performance monitoring, and fine-tuning.

---

## 🧪 Testing

Run unit tests with:
```bash
pytest tests/
```

---

## 🛠️ Tech Stack
- **LangChain** – Orchestration framework for LLMs
- **Groq LLM** – Fast inference large language models
- **Qdrant** – Vector database for embeddings
- **Weather API** – Live weather data integration
- **Streamlit** – Frontend interface (optional)
- **LangSmith** – Tracing, debugging, and evaluation

---

## 📌 Future Improvements
- Add support for multiple APIs (e.g., finance, news)
- Enhance RAG pipeline with rerankers
- Add Docker for containerized deployment

---

## 👨‍💻 Author
Built by **Saransh Dhage**  
📍 Data Scientist | AI/ML Enthusiast | Exploring Finance & AI

---
