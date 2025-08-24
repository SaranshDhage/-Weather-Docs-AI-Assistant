from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from settings import settings
from embeddings import get_embeddings

# Try both import paths for compatibility
try:
    from langchain_qdrant import QdrantVectorStore as Qdrant
except ImportError:
    from langchain_community.vectorstores import Qdrant  # type: ignore


def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client with URL + API key."""
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None
    )


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = 384):
    """Ensure a Qdrant collection exists with the proper vector size."""
    existing_collections = [
        c.name for c in client.get_collections().collections]
    if collection_name not in existing_collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size, distance=Distance.COSINE)
        )


def ingest_pdf_to_qdrant(pdf_path: str, collection: Optional[str] = None) -> int:
    """Load a PDF, split into chunks, embed, and insert into Qdrant."""
    collection = collection or settings.docs_collection
    loader = PyPDFLoader(pdf_path)
    pages: List[Document] = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(pages)

    embeddings = get_embeddings()
    client = get_qdrant_client()
    ensure_collection(
        client,
        collection,
        vector_size=embeddings.client.get_sentence_embedding_dimension()  # type: ignore
    )

    Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        collection_name=collection
    )
    return len(chunks)


def get_vectorstore(collection: Optional[str] = None):
    """Return a Qdrant-backed vectorstore for a given collection."""
    collection = collection or settings.docs_collection
    embeddings = get_embeddings()
    return Qdrant(
        client=get_qdrant_client(),
        collection_name=collection,
        embedding=embeddings   # ✅ fixed
    )


def upsert_interaction(summary_text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Insert or update a single interaction document in Qdrant."""
    embeddings = get_embeddings()
    client = get_qdrant_client()
    ensure_collection(
        client,
        settings.interactions_collection,
        vector_size=embeddings.client.get_sentence_embedding_dimension()  # type: ignore
    )

    vs = Qdrant(
        client=client,
        collection_name=settings.interactions_collection,
        embedding=embeddings   # ✅ fixed
    )

    doc = Document(page_content=summary_text, metadata=metadata or {})
    ids = vs.add_documents([doc])
    return ids[0] if ids else ""
