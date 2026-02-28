import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.core.config import MAX_FILE_MB, VECTORDB_DIR
from app.llm.embeddings_factory import get_embeddings
from app.services.retrieval_service import build_chain
from app.session.manager import set_session, get_session

logger = logging.getLogger(__name__)

def ingest_document(file_path: str, filename: str, session_id: str):
    try:
        path = Path(file_path)

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            yield f"❌ File too large ({size_mb:.1f} MB). Max: {MAX_FILE_MB} MB"
            return

        ext = path.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            yield f"❌ Unsupported type: '{ext}'. Use PDF, DOCX, or TXT."
            return

        docs = loader.load()
        if not docs:
            yield "❌ File is empty or unreadable."
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        if not chunks:
            yield "❌ No text extracted. Scanned PDF? Try a text-based one."
            return

        yield f"🧠 Generating embeddings (HuggingFace)..."
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        yield "💾 Saving index to secure storage..."
        vdb_path = VECTORDB_DIR / session_id
        vectorstore.save_local(str(vdb_path))

        yield "⚙️ Configuring AI reasoning engine..."
        chain = build_chain(vectorstore)

        set_session(session_id, {
            "chain":        chain,
            "vectorstore":  vectorstore,
            "chat_history": [],
            "filename":     filename,
            "chunks":       len(chunks),
        })
        
        yield "✅ Document ready! You can now ask questions."

    except Exception as e:
        logger.exception(f"Ingestion fail for {filename}")
        yield f"❌ Error: {str(e)}"

def restore_session(session_id: str) -> bool:
    vdb_path = VECTORDB_DIR / session_id
    if not vdb_path.exists():
        return False
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(
            str(vdb_path), embeddings, allow_dangerous_deserialization=True
        )
        chain = build_chain(vectorstore)
        
        set_session(session_id, {
            "chain":        chain,
            "vectorstore":  vectorstore,
            "chat_history": [],
            "filename":     "restored",
            "chunks":       0,
        })
        return True
    except Exception as e:
        logger.warning(f"restore_session failed: {e}")
        return False
