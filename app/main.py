"""
Enhanced Document Q&A System
✅ LCEL Pipeline + Real Streaming + Language Detection + Session Persistence
"""

import os
import uuid
import time
import shutil
import logging
import threading
from pathlib import Path
from typing import Dict
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))   # ensure app/ on path
from evaluator import SystemEvaluator
_evaluator = SystemEvaluator()
EVAL_REPORT_PATH = "eval_report.json"

# ─────────────────────────────────────────
# Storage
# ─────────────────────────────────────────
UPLOAD_DIR   = Path("data/uploads")
VECTORDB_DIR = Path("data/vectordb")
CACHE_DIR    = Path("data/cache")
MAX_FILE_MB  = 50
SESSION_TIMEOUT_HOURS = 2

for d in [UPLOAD_DIR, VECTORDB_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sessions: Dict = {}
sessions_lock = threading.Lock()

# ─────────────────────────────────────────
# Imports
# ─────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import re

load_dotenv(Path(__file__).parent.parent / ".env")

# ─────────────────────────────────────────
# Math Notation Fix
# ─────────────────────────────────────────
def fix_math_notation(text: str) -> str:
    """
    Convert LLM math output to proper LaTeX dollar-sign delimiters so
    Gradio's KaTeX renderer can display them:
      - ((\\expr))           →  $$\\expr$$   (display block)
      - (\\LaTeXCommand ...) →  $\\expr$     (inline)
      - Bare Unicode symbols →  $\\latex$    (inline, see table below)
    """
    # ── Step 1: (( ... )) → $$ ... $$ ─────────────────────────────────
    text = re.sub(r'\(\(([^()]+)\)\)', r'$$\1$$', text)

    # ── Step 2: ( \cmd ... ) → $ \cmd ... $ ────────────────────────────
    # Only when no $ already present (avoid double-wrapping)
    if '$' not in text:
        text = re.sub(
            r'\(([^()]*\\[a-zA-Z{][^()]*)\)',
            r'$\1$',
            text
        )

    # ── Step 3: Unicode math symbols → LaTeX (outside $...$ blocks) ────
    _UNICODE_TO_LATEX = [
        # Relations
        ('\u2264', r'$\le$'),     ('\u2265', r'$\ge$'),
        ('\u2260', r'$\neq$'),    ('\u2248', r'$\approx$'),
        ('\u2261', r'$\equiv$'),  ('\u223c', r'$\sim$'),
        ('\u2243', r'$\simeq$'),  ('\u2245', r'$\cong$'),
        # Set symbols
        ('\u2208', r'$\in$'),     ('\u2209', r'$\notin$'),
        ('\u2286', r'$\subseteq$'), ('\u2287', r'$\supseteq$'),
        ('\u2282', r'$\subset$'), ('\u2283', r'$\supset$'),
        ('\u2205', r'$\varnothing$'),
        ('\u222a', r'$\cup$'),    ('\u2229', r'$\cap$'),
        ('\u2216', r'$\setminus$'),
        # Logic
        ('\u2200', r'$\forall$'), ('\u2203', r'$\exists$'),
        ('\u00ac', r'$\neg$'),    ('\u2227', r'$\wedge$'),
        ('\u2228', r'$\vee$'),    ('\u21d2', r'$\Rightarrow$'),
        ('\u21d4', r'$\Leftrightarrow$'), ('\u2192', r'$\to$'),
        # Arithmetic / calculus
        ('\u2211', r'$\sum$'),    ('\u220f', r'$\prod$'),
        ('\u222b', r'$\int$'),    ('\u221e', r'$\infty$'),
        ('\u00b1', r'$\pm$'),     ('\u00d7', r'$\times$'),
        ('\u00f7', r'$\div$'),    ('\u00b7', r'$\cdot$'),
        ('\u221a', r'$\sqrt{}$'), ('\u2032', r"$'$"),
        # Greek (common standalone uses outside $)
        ('\u03b1', r'$\alpha$'),  ('\u03b2', r'$\beta$'),
        ('\u03b3', r'$\gamma$'),  ('\u03b4', r'$\delta$'),
        ('\u03b5', r'$\epsilon$'),('\u03bb', r'$\lambda$'),
        ('\u03bc', r'$\mu$'),     ('\u03c0', r'$\pi$'),
        ('\u03c1', r'$\rho$'),    ('\u03c3', r'$\sigma$'),
        ('\u03c6', r'$\phi$'),    ('\u03c8', r'$\psi$'),
        ('\u03c9', r'$\omega$'),  ('\u0394', r'$\Delta$'),
        ('\u03a3', r'$\Sigma$'),  ('\u03a9', r'$\Omega$'),
    ]

    def _replace_outside_math(s: str) -> str:
        # Split on existing $...$ blocks so we never touch them
        parts = re.split(r'(\$\$[\s\S]*?\$\$|\$[^$\n]+?\$)', s)
        result = []
        for i, part in enumerate(parts):
            if i % 2 == 1:      # inside existing $...$ — leave alone
                result.append(part)
            else:               # plain text — apply symbol replacements
                for sym, latex in _UNICODE_TO_LATEX:
                    part = part.replace(sym, latex)
                result.append(part)
        return ''.join(result)

    text = _replace_outside_math(text)
    return text


# ─────────────────────────────────────────
# Language Detection
# ─────────────────────────────────────────
def detect_language(text: str) -> str:
    """Returns 'ar' if Arabic, 'en' otherwise."""
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    ratio = arabic_chars / max(len(text.strip()), 1)
    return "ar" if ratio > 0.2 else "en"


def get_language_instruction(lang: str) -> str:
    if lang == "ar":
        return "يجب أن تكون إجابتك باللغة العربية فقط."
    return "You must respond in English only."


# ─────────────────────────────────────────
# Intent Detection — is this a doc question or casual chat?
# ─────────────────────────────────────────

# Keywords that strongly suggest a document question
DOC_KEYWORDS = {
    # Arabic
    "ما", "من", "كيف", "متى", "أين", "لماذا", "هل", "اشرح", "وضح",
    "اذكر", "عرّف", "ما هو", "ما هي", "ملخص", "خلاصة", "نتيجة",
    "مستند", "وثيقة", "ملف", "تقرير", "بحث", "دراسة", "فقرة", "صفحة",
    # English
    "what", "who", "when", "where", "why", "how", "explain", "describe",
    "summarize", "summary", "define", "list", "document", "report",
    "according", "mention", "states", "says", "page", "chapter", "section",
}

CASUAL_PATTERNS = [
    # Greetings
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "good night", "howdy", "sup", "what's up",
    # Arabic greetings
    "مرحبا", "السلام", "اهلا", "أهلا", "صباح", "مساء", "هاي", "هلو",
    "كيف حالك", "كيف الحال", "عامل ايه", "ازيك", "ازيكم",
    # Farewell
    "bye", "goodbye", "see you", "later", "take care",
    "وداعا", "مع السلامة", "باي", "إلى اللقاء",
    # Thanks
    "thanks", "thank you", "thx", "ty",
    "شكرا", "شكراً", "ممنون", "متشكر",
    # Praise / reactions
    "great", "awesome", "nice", "cool", "perfect", "wow", "ok", "okay",
    "ممتاز", "رائع", "حلو", "تمام", "عظيم", "أوكي",
]


def is_casual_message(text: str) -> bool:
    """
    Returns True ONLY for clear greetings, farewells, and thanks.
    Does NOT classify short questions as casual.
    """
    clean = text.strip().lower().rstrip("!.#(),?؟")
    for pattern in CASUAL_PATTERNS:
        if clean == pattern or clean.startswith(pattern + " ") or clean.startswith(pattern + "،"):
            return True
    return False


# ─────────────────────────────────────────
# LLM
# ─────────────────────────────────────────
def get_llm(streaming: bool = True):
    model_name = os.getenv("OPENROUTER_MODEL")
    url = os.getenv("base_url")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    return ChatOpenAI(
        model=model_name,
        base_url=url,
        api_key=api_key,
        temperature=0,
        streaming=streaming,
    )


# ─────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# Casual Chat Chain (no document needed)
# ─────────────────────────────────────────
def build_casual_chain():

    """Simple chain for greetings and small talk — no retrieval."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly and helpful assistant.
                   Respond naturally and warmly to the user.
                   {language_instruction}
                   Keep your reply short and friendly."""),

        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def add_language(inputs: dict) -> dict:
        lang = detect_language(inputs["question"])
        return {
            **inputs,
            "language_instruction": get_language_instruction(lang),
        }

    chain = (
        RunnableLambda(add_language)
        | RunnableLambda(lambda x: {
            "language_instruction": x["language_instruction"],
            "chat_history": x.get("chat_history", []),
            "question": x["question"],
        })
        | prompt
        | get_llm(streaming=True)
        | StrOutputParser()
    )
    return chain


_casual_chain = None

def get_casual_chain():
    global _casual_chain
    if _casual_chain is None:
        _casual_chain = build_casual_chain()
    return _casual_chain


# ─────────────────────────────────────────
# Build LCEL Chain
# ─────────────────────────────────────────
def build_chain(vectorstore: FAISS):
    """
    LCEL chain that:
    1. Detects language from the question
    2. Retrieves relevant docs via MMR
    3. Formats prompt with language instruction
    4. Streams response token by token
    """

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "lambda_mult": 0.7},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a document assistant. You ONLY answer based on the document context provided below.

STRICT RULES — follow them without exception:
1. Use ONLY the information in the Document Context below to answer.
2. Do NOT use any outside knowledge, general knowledge, or training data.
3. If the answer is not found in the Document Context, respond with:
   - Arabic: "لم أجد هذه المعلومة في المستند المرفوع."
   - English: "This information is not available in the uploaded document."
4. Never guess, infer, or make up information.
5. Mention page numbers when available (e.g., [Page 3]).
6. {language_instruction}

FORMATTING RULES — always apply:
- Use **bold** for key terms and important phrases.
- Use bullet points (- item) for lists.
- Use numbered lists (1. 2. 3.) for steps or ordered items.
- Use `## Heading` for sections when the answer has multiple parts.
- Use tables (| col | col |) when comparing items or presenting structured data.
- Keep paragraphs short and readable. Add a blank line between sections.
- Do NOT write long dense paragraphs — break them up.

MATH FORMATTING — CRITICAL, follow exactly:
- ALL math expressions MUST use LaTeX dollar-sign delimiters.
- Inline math  → wrap with single $: example $\\mathbb{{Z}}$, $\\le$, $\\rho \\circ \\sigma$
- Display math → wrap with double $$: example $$\\sum_{{i=1}}^{{n}} x_i$$
- NEVER use bare Unicode symbols like ≤ ≥ ∈ ∉ ⊆ ∅ ∪ ∩ ∀ ∃ — use LaTeX ONLY.
- NEVER use parentheses like (\\mathbb Z) or ((\\rho)) for math — use $ or $$ ONLY.
- NEVER write math symbols as plain text.
- Examples of CORRECT format:
  * $\\mathbb{{Z}}$ not (\\mathbb Z)
  * $(\\mathbb{{Z}}, \\le)$ not ((\\mathbb Z, \\le))
  * $\\rho \\circ \\sigma$ not (\\rho\\circ\\sigma)
  * $\\mathcal{{P}}(A)$ not (\\mathcal P(A))
  * $\\varnothing$ not ∅ and not (\\varnothing)
  * $\\le$ not ≤
  * $\\in$ not ∈
  * $\\subseteq$ not ⊆

Document Context:
{context}
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs) -> str:
        parts = []
        for doc in docs:
            page = doc.metadata.get("page", "?")
            parts.append(f"[Page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    # Step 1: add language info
    def add_language(inputs: dict) -> dict:
        lang = detect_language(inputs["question"])
        return {
            **inputs,
            "language_instruction": get_language_instruction(lang),
            "detected_lang": lang,
        }

    # Step 2: retrieve docs and add to inputs
    def retrieve(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["question"])
        return {
            **inputs,
            "context": format_docs(docs),
            "source_docs": docs,
        }

    # Step 3: build prompt inputs (only what prompt expects)
    def build_prompt_inputs(inputs: dict) -> dict:
        return {
            "language_instruction": inputs["language_instruction"],
            "context": inputs["context"],
            "chat_history": inputs.get("chat_history", []),
            "question": inputs["question"],
        }

    # LCEL answer chain with streaming
    answer_chain = (
        RunnableLambda(build_prompt_inputs)
        | prompt
        | get_llm(streaming=True)
        | StrOutputParser()
    )

    # Full chain: language → retrieve → { answer + source_docs }
    chain = (
        RunnableLambda(add_language)
        | RunnableLambda(retrieve)
        | {
            "answer":      answer_chain,
            "source_docs": RunnableLambda(lambda x: x["source_docs"]),
            "detected_lang": RunnableLambda(lambda x: x["detected_lang"]),
        }
    )

    return chain


# ─────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────
def ingest_document(file_path: str, filename: str, session_id: str):
    try:
        path = Path(file_path)

        # File size check
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            return False, f"❌ File too large ({size_mb:.1f} MB). Max: {MAX_FILE_MB} MB"

        # Load by type
        ext = path.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            return False, f"❌ Unsupported type: '{ext}'. Use PDF, DOCX, or TXT."

        docs = loader.load()
        if not docs:
            return False, "❌ File is empty or unreadable."

        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # Build vectorstore
        embeddings = get_embeddings()
        if not chunks:
            yield "❌ No text extracted. Scanned PDF? Try a text-based one."
            return

        # Build vectorstore
        yield f"🧠 Generating embeddings (HuggingFace)..."
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Persist to disk
        yield "💾 Saving index to secure storage..."
        vdb_path = VECTORDB_DIR / session_id
        vectorstore.save_local(str(vdb_path))

        # Build LCEL chain
        yield "⚙️ Configuring AI reasoning engine..."
        chain = build_chain(vectorstore)

        with sessions_lock:
            sessions[session_id] = {
                "chain":        chain,
                "vectorstore":  vectorstore,
                "chat_history": [],
                "filename":     filename,
                "chunks":       len(chunks),
                "last_used":    datetime.now(),
            }
        
        yield "✅ Document ready! You can now ask questions."

    except Exception as e:
        logger.exception(f"Ingestion fail for {filename}")
        yield f"❌ Error: {str(e)}"


# ─────────────────────────────────────────
# Restore session from disk
# ─────────────────────────────────────────
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
        with sessions_lock:
            sessions[session_id] = {
                "chain":        chain,
                "vectorstore":  vectorstore,
                "chat_history": [],
                "filename":     "restored",
                "chunks":       0,
                "last_used":    datetime.now(),
            }
        return True
    except Exception as e:
        logger.warning(f"restore_session failed: {e}")
        return False


# ─────────────────────────────────────────
# Session Cleanup
# ─────────────────────────────────────────
def cleanup_old_sessions():
    cutoff = datetime.now() - timedelta(hours=SESSION_TIMEOUT_HOURS)
    with sessions_lock:
        expired = [
            sid for sid, data in sessions.items()
            if data.get("last_used", datetime.now()) < cutoff
        ]
        for sid in expired:
            sessions.pop(sid, None)
            vdb_path = VECTORDB_DIR / sid
            if vdb_path.exists():
                shutil.rmtree(vdb_path, ignore_errors=True)
            logger.info(f"🧹 Cleaned session: {sid}")


def start_cleanup_thread():
    def _loop():
        while True:
            time.sleep(1800)
            try:
                cleanup_old_sessions()
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")
    threading.Thread(target=_loop, daemon=True).start()


# ─────────────────────────────────────────
# ✅ Streaming QA Generator
# ─────────────────────────────────────────
def ask_question_stream(question: str, session_id: str):
    """
    Generator that yields tokens.
    - Greetings/thanks  → casual chain (no document needed)
    - Everything else   → strict RAG chain (document only)
    """
    if not question.strip():
        yield "⚠️ Question is empty."
        return

    # ── Casual: greetings / thanks / farewells ────────────────
    if is_casual_message(question):
        chat_history = []
        if session_id in sessions:
            with sessions_lock:
                chat_history = list(sessions[session_id].get("chat_history", []))
        casual = get_casual_chain()
        full_answer = ""
        for token in casual.stream({"question": question, "chat_history": chat_history}):
            if token:
                full_answer += token
                yield token
        if session_id in sessions:
            with sessions_lock:
                sessions[session_id]["chat_history"].append(HumanMessage(content=question))
                sessions[session_id]["chat_history"].append(AIMessage(content=full_answer))
        yield {"__sources__": []}
        return

    # ── Document RAG ─────────────────────────────────
    if session_id not in sessions:
        if not restore_session(session_id):
            yield "⚠️ Please upload a document first before asking questions."
            return

    with sessions_lock:
        session = sessions[session_id]
        session["last_used"] = datetime.now()
        chain        = session["chain"]
        chat_history = list(session["chat_history"])

    inputs = {
        "question":     question,
        "chat_history": chat_history,
    }

    full_answer = ""
    source_docs = []

    try:
        for chunk in chain.stream(inputs):
            if "answer" in chunk:
                token = chunk["answer"]
                if token:
                    full_answer += token
                    yield token
            if "source_docs" in chunk:
                source_docs = chunk["source_docs"]
        yield {"__sources__": source_docs}

        with sessions_lock:
            history = sessions[session_id]["chat_history"]
            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=full_answer))
            if len(history) > 20:
                sessions[session_id]["chat_history"] = history[-20:]

    except Exception as e:
        logger.exception("ask_question_stream error")
        yield f"\n\n❌ Error: {str(e)}"



# ─────────────────────────────────────────
# UI Helpers

# ─────────────────────────────────────────
def handle_upload(file, session_id):
    if file is None:
        yield "⚠️ No file selected."
        return
    filename = Path(file.name).name
    dest = UPLOAD_DIR / f"{session_id}_{filename}"
    try:
        shutil.copy(file.name, dest)
    except Exception as e:
        yield f"❌ Failed to copy '{filename}': {e}"
        return
    
    for status in ingest_document(str(dest), filename, session_id):
        yield status



def handle_chat(msg: str, history, session_id: str):
    """Gradio streaming generator for the chat tab."""
    if not msg.strip():
        yield history, "", ""
        return

    history = history or []
    history.append([msg, ""])
    yield history, "", "" # First yield to show the user's message

    sources_text = ""
    start = time.time()
    full_answer = ""

    for chunk in ask_question_stream(msg, session_id):
        if isinstance(chunk, dict) and "__sources__" in chunk:
            # Build citations box
            seen = set()
            lines = []
            for doc in chunk["__sources__"]:
                page    = doc.metadata.get("page", "?")
                snippet = doc.page_content[:150].replace("\n", " ").strip()
                key     = (page, snippet[:40])
                if key not in seen:
                    seen.add(key)
                    lines.append(f"📄 Page {page}: {snippet}...")
            latency      = round(time.time() - start, 2)
            sources_text = "\n\n".join(lines) + f"\n\n⏱ {latency}s"

            # ── Auto-evaluate ──
            has_sources = bool(chunk["__sources__"])
            if has_sources and full_answer.strip():
                _STOPWORDS = {"what", "does", "talk", "about", "this", "that", "document", "الملف", "الوثيقة"}
                import re as _re
                keywords = [_re.sub(r'[^\w]', '', w).lower() for w in full_answer.split() if len(w) > 4][:10]
                _evaluator.evaluate_answer(msg, full_answer, keywords, latency)
                _evaluator.save_report(EVAL_REPORT_PATH)
        else:
            full_answer += chunk
            history[-1][1] = fix_math_notation(full_answer)
            yield history, "", sources_text

    yield history, "", sources_text


def get_eval_report() -> str:
    """Build evaluation summary text for the Gradio panel."""
    if not _evaluator.results:
        return "⚠️There are no evaluation yet"

    n        = len(_evaluator.results)
    avg_cov  = sum(r['keyword_coverage'] for r in _evaluator.results) / n
    avg_lat  = sum(r['latency_seconds']  for r in _evaluator.results) / n
    avg_len  = sum(r['answer_length']    for r in _evaluator.results) / n
    lat_ok   = sum(1 for r in _evaluator.results if r['latency_ok'])
    threshold = _evaluator.LATENCY_THRESHOLD

    lines = [
        f"📊 إجمالي الأسئلة       : {n}",
        f"✅ متوسط تغطية الكلمات : {avg_cov:.1f}%",
        f"📝 متوسط طول الإجابة   : {avg_len:.0f} حرف",
        f"⏱ متوسط زمن الاستجابة : {avg_lat:.2f}s  (الحد: {threshold}s)",
        f"{'✅' if lat_ok == n else '⚠️'} ضمن الحد الزمني       : {lat_ok}/{n}",
        "─" * 45,
        "🔎 آخر 5 أسئلة:",
    ]
    for i, r in enumerate(_evaluator.results[-5:], 1):
        lat_icon = "✅" if r['latency_ok'] else "⚠️"
        lines.append(
            f"\n[{i}] ❓ {r['question'][:60]}…"
            f"\n    تغطية: {r['keyword_coverage']}%  |  "
            f"زمن: {r['latency_seconds']}s {lat_icon}  |  "
            f"طول: {r['answer_length']} حرف  |  "
            f"مفقود: {r['missing_keywords'] or 'لا شيء'}"
        )
    return "\n".join(lines)



def get_session_info(session_id: str) -> str:
    with sessions_lock:
        session = sessions.get(session_id)
    if session:
        turns = len(session.get("chat_history", [])) // 2
        return (
            f"✅  File: {session.get('filename')}  |  "
            f"Chunks: {session.get('chunks')}  |  "
            f"History: {turns} turns"
        )
    if (VECTORDB_DIR / session_id).exists():
        return "📂 Saved document found — will load on first question"
    return "❌ No document uploaded yet"


def clear_history(session_id: str):
    with sessions_lock:
        if session_id in sessions:
            sessions[session_id]["chat_history"] = []
    return [], ""



# ─────────────────────────────────────────
# Custom CSS & Identity
# ─────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
    --bg-dark: #020617;
    --card-bg: rgba(30, 41, 59, 0.7);
    --border-color: rgba(255, 255, 255, 0.1);
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
}

body, .gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--text-main) !important;
}

.main-header {
    text-align: center;
    padding: 2rem 0;
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    font-size: 2.5rem !important;
}

.glass-card {
    background: var(--card-bg) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5) !important;
}

.chat-bubble {
    border-radius: 12px !important;
    border: 1px solid var(--border-color) !important;
    background: rgba(15, 23, 42, 0.5) !important;
}

.action-btn {
    background: var(--primary-gradient) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}

.action-btn:hover {
    transform: scale(1.02);
    filter: brightness(1.1);
    box-shadow: 0 0 20px rgba(129, 140, 248, 0.4) !important;
}

/* Fix for light mode text persistence in some components */
.gradio-container label, .gradio-container span, .gradio-container p {
    color: var(--text-muted) !important;
}

h1, h2, h3 {
    color: var(--text-main) !important;
}

footer { visibility: hidden !important; }
"""

# ─────────────────────────────────────────
# Gradio UI  
# ─────────────────────────────────────────
import gradio as gr


def build_ui():
    theme = gr.themes.Soft(
        spacing_size="sm",
        radius_size="lg",
        primary_hue="indigo",
        neutral_hue="slate",
    ).set(
        body_background_fill="#020617",
        block_background_fill="#0f172a",
        block_border_width="1px",
        input_background_fill="#1e293b",
    )

    with gr.Blocks(title="Document AI Assistant", theme=theme, css=CUSTOM_CSS) as demo:
        session_id = gr.State(str(__import__("uuid").uuid4()))
        last_msg = gr.State("")

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h1 class='main-header'>🧠 Document AI Assistant</h1>")
                gr.Markdown(
                    "<p style='text-align: center; color: #94a3b8; margin-top: -10px;'>"
                    "Intelligent RAG system for instant document Q&A"
                    "</p>"
                )

        with gr.Row(equal_height=True):
            # ── LEFT: Chat Interface ─────────────────────────────
            with gr.Column(scale=3):
                with gr.Group(elem_classes="glass-card"):
                    chatbot = gr.Chatbot(
                        height=550,
                        show_label=False,
                        bubble_full_width=False,
                        placeholder="✨ Hello! Upload a document to start exploring.",
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": "$",  "right": "$",  "display": False},
                        ],
                        render_markdown=True,
                        show_copy_button=True,
                        elem_classes="chat-bubble"
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask me anything...",
                            label="", show_label=False,
                            scale=8, 
                            container=False,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1, elem_classes="action-btn")

                with gr.Row():
                    gr.Markdown("💡 **Suggestions:**")
                    sug1 = gr.Button("Summarize the document", size="sm", variant="secondary")
                    sug2 = gr.Button("What are the key findings?", size="sm", variant="secondary")
                    sug3 = gr.Button("Identify risks or gaps", size="sm", variant="secondary")

                with gr.Accordion("📚 Sources & References", open=False):
                    sources_box = gr.Textbox(
                        label="", show_label=False,
                        lines=3, interactive=False,
                        placeholder="Sources will appear here..."
                    )
                
                with gr.Row():
                    clear_btn = gr.Button("✕ Clear History", size="sm", variant="link")
                    regen_btn = gr.Button("🔄 Regenerate", size="sm", variant="link")

                # Events
                def chat_wrapper(msg, history, sid):
                    for h, m, s in handle_chat(msg, history, sid):
                        yield h, m, s, msg

                submit_event = send_btn.click(
                    chat_wrapper,
                    inputs=[msg_input, chatbot, session_id],
                    outputs=[chatbot, msg_input, sources_box, last_msg],
                )
                msg_input.submit(
                    chat_wrapper,
                    inputs=[msg_input, chatbot, session_id],
                    outputs=[chatbot, msg_input, sources_box, last_msg],
                )

                def regenerate_wrapper(history, sid, last_m):
                    if not last_m or not history:
                        yield history, "", ""
                        return
                    # Remove last AI message
                    if len(history) > 0 and history[-1][1]:
                        history[-1][1] = ""
                    for h, m, s in handle_chat(last_m, history[:-1], sid):
                        yield h, m, s

                regen_btn.click(
                    regenerate_wrapper,
                    inputs=[chatbot, session_id, last_msg],
                    outputs=[chatbot, msg_input, sources_box],
                )

                clear_btn.click(fn=clear_history, inputs=session_id, outputs=[chatbot, sources_box])

                for sug in [sug1, sug2, sug3]:
                    sug.click(chat_wrapper, inputs=[sug, chatbot, session_id], outputs=[chatbot, msg_input, sources_box, last_msg])

            # ── RIGHT: Controls ──────────────────────────────────
            with gr.Column(scale=1):
                with gr.Group(elem_classes="glass-card"):
                    gr.Markdown("### 📂 Data Ingestion")
                    file_input = gr.File(
                        label="",
                        file_types=[".pdf", ".docx", ".doc", ".txt"],
                    )
                    upload_btn    = gr.Button("⬆ Process Document", variant="primary", elem_classes="action-btn")
                    upload_status = gr.Textbox(label="Status", interactive=False, lines=1)
                
                with gr.Group(elem_classes="glass-card"):
                    gr.Markdown("### ℹ️ Session Info")
                    info_box = gr.Textbox(label="", show_label=False, interactive=False)

                upload_btn.click(
                    handle_upload, inputs=[file_input, session_id], outputs=upload_status
                ).then(
                    get_session_info, inputs=session_id, outputs=info_box
                )

                with gr.Accordion("📊 System Performance", open=False):
                    eval_refresh_btn = gr.Button("🔄 Update Metrics", variant="secondary")
                    eval_box = gr.Textbox(label="", show_label=False, lines=10, interactive=False)
                    eval_refresh_btn.click(fn=get_eval_report, inputs=[], outputs=eval_box)

    return demo


# ─────────────────────────────────────────
if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    print("🚀 Starting smart contract assistant ")
    start_cleanup_thread()
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,
    )