"""
Backend API Layer - REST API
يوفر endpoints لإدارة الملفات والشات
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid
import time
import shutil
from pathlib import Path

app = FastAPI(
    title="Document Q&A API",
    description="نظام أسئلة وأجوبة للمستندات",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# Models
# ─────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    latency: float
    session_id: str

class StatusResponse(BaseModel):
    status: str
    document: Optional[str]
    message: str

# ─────────────────────────────────────────
# File API
# ─────────────────────────────────────────
@app.post("/upload", tags=["Files"])
async def upload_file(file: UploadFile = File(...)):
    """رفع ملف جديد"""
    allowed = [".pdf", ".docx", ".doc", ".txt"]
    ext = Path(file.filename).suffix.lower()
    
    if ext not in allowed:
        raise HTTPException(400, f"نوع الملف غير مدعوم: {ext}")
    
    # Save file
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    return {
        "filename": file.filename,
        "size_bytes": dest.stat().st_size,
        "message": "تم الرفع بنجاح - ابدأ المعالجة عبر /process"
    }

@app.post("/process", tags=["Files"])
async def process_file(filename: str):
    """معالجة ملف مرفوع"""
    from app.main import ingest_document
    
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "الملف غير موجود")
    
    success, msg = ingest_document(str(file_path), filename)
    if not success:
        raise HTTPException(500, msg)
    
    return {"status": "ready", "message": msg}

@app.get("/status", response_model=StatusResponse, tags=["Files"])
async def get_status():
    """الحالة الحالية للنظام"""
    from app.main import processing_status, current_doc_name, get_status as _get_status
    
    return StatusResponse(
        status=processing_status,
        document=current_doc_name,
        message=_get_status()
    )

# ─────────────────────────────────────────
# Chat API
# ─────────────────────────────────────────
@app.post("/ask", response_model=AskResponse, tags=["Chat"])
async def ask(request: AskRequest):
    """طرح سؤال على المستند"""
    from app.main import ask_question
    
    session_id = request.session_id or str(uuid.uuid4())
    start = time.time()
    
    answer, sources = ask_question(request.question, session_id)
    latency = round(time.time() - start, 2)
    
    return AskResponse(
        answer=answer,
        sources=sources,
        latency=latency,
        session_id=session_id
    )

@app.get("/history/{session_id}", tags=["Chat"])
async def get_history(session_id: str):
    """سجل المحادثة"""
    from app.main import sessions
    
    history = sessions.get(session_id, [])
    return {"session_id": session_id, "messages": history}

@app.delete("/session/{session_id}", tags=["Chat"])
async def delete_session(session_id: str):
    """حذف جلسة"""
    from app.main import sessions
    
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "تم حذف الجلسة"}
    return {"message": "الجلسة غير موجودة"}

@app.get("/", tags=["Health"])
async def root():
    return {"status": "running", "message": "Document Q&A API يعمل بنجاح 🚀"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
