from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from settings import settings

def get_embeddings() -> HuggingFaceEmbeddings:
    # You can swap this for other providers if desired.
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
