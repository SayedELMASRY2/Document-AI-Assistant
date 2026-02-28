from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import LocalFileStore

from app.core.config import CACHE_DIR

_embeddings_instance = None

def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        base = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        store = LocalFileStore(str(CACHE_DIR))
        _embeddings_instance = CacheBackedEmbeddings.from_bytes_store(base, store)
    return _embeddings_instance
