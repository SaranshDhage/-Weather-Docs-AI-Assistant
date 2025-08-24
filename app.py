from settings import settings
from vectorstore import ingest_pdf_to_qdrant
from graph import build_graph
from typing import Optional
import streamlit as st
from dotenv import load_dotenv
import tempfile
load_dotenv()


st.set_page_config(
    page_title="Weather & Docs AI Assistant",
    page_icon="⛅",
    layout="wide"
)

st.title("⛅📄 Weather & Docs AI Assistant")

# ---- Sidebar: PDF ingestion ----
with st.sidebar:
    st.header("Setup & Ingestion")
    pdf = st.file_uploader("Upload a PDF to ingest", type=["pdf"])
    if st.button("Ingest PDF", type="primary") and pdf is not None:
        # Save uploaded file safely (works cross-platform)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            tmp_path = tmp_file.name

        # Call vectorstore ingestion
        n = ingest_pdf_to_qdrant(tmp_path, collection=settings.docs_collection)
        st.success(
            f"✅ Ingested {n} chunks into Qdrant collection '{settings.docs_collection}'.")

    st.divider()
    st.subheader("Config (read-only)")
    st.write(f"Groq model: `{settings.groq_model}`")
    st.write(f"Qdrant URL: `{settings.qdrant_url}`")
    st.write(f"Docs collection: `{settings.docs_collection}`")
    st.write(f"Interactions collection: `{settings.interactions_collection}`")

# ---- Initialize graph & history ----
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "history" not in st.session_state:
    st.session_state.history = []

# ---- Chat Interface ----
query = st.chat_input(
    "Ask about weather (e.g., 'weather in Mumbai') or your PDF...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Thinking..."):
        try:
            # LangGraph pipeline (router decides weather vs RAG)
            res = st.session_state.graph.invoke({"query": query, "meta": {}})
            answer = res.get("answer", "(No answer)")
        except Exception as e:
            answer = f"❌ Error: {e}"

    st.session_state.history.append((query, answer))

    with st.chat_message("assistant"):
        st.write(answer)

# ---- Show Conversation History ----
if st.session_state.history:
    with st.expander("Conversation History"):
        for u, a in st.session_state.history[-10:]:
            st.markdown(f"**You:** {u}\n\n**Assistant:** {a}")
