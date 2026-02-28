# 📚 Document Q&A System
> Intelligent Q&A system based on RAG (Retrieval-Augmented Generation) with streaming support and auto language detection.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend Layer                       │
│              Gradio UI — app/main.py (Port 7860)        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 LLM Pipeline Layer                      │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  Ingestion  │  │  Retrieval   │  │    Answer Gen  │  │
│  │  Pipeline   │  │  Pipeline    │  │    Pipeline    │  │
│  │             │  │              │  │                │  │
│  │ Load → Split│  │ Embed Query  │  │ Build Prompt   │  │
│  │ → Embed     │  │ → MMR Search │  │ → Stream LLM   │  │
│  │ → Store     │  │ → Top-K      │  │ → Citations    │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   Storage Layer                         │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │File Storage │  │  Vector DB   │  │  Chat Memory   │  │
│  │data/uploads │  │    FAISS     │  │  RAM + Session │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.10+
- At least 4GB RAM
- [OpenRouter API](https://openrouter.ai) Key (Free tier available)

### Steps

**Quick Setup (Windows):**
```bat
setup.bat
```
The script will automate everything: venv creation, installing dependencies, creating directories, and generating `.env`.

**Or Manually:**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup environment variables
copy .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# 3. Run the UI
python -m app.main
```

Open your browser at: **http://localhost:7860**

---

## 📁 Project Structure

```
project/
├── app/
│   ├── core/
│   │   ├── config.py             # Constants and configs
│   │   ├── locks.py              # Threading locks
│   │   └── exceptions.py         # Custom application exceptions
│   │
│   ├── services/
│   │   ├── chat_service.py       # Handles chat messages and logic
│   │   ├── ingestion_service.py  # Handles document processing
│   │   └── retrieval_service.py  # LangChain pipelines and retrieval
│   │
│   ├── llm/
│   │   ├── llm_factory.py        # Chat LLM initialization (OpenRouter)
│   │   └── embeddings_factory.py # Embeddings instantiation (HuggingFace)
│   │
│   ├── evaluation/
│   │   ├── metrics.py            # Answer evaluation formulas
│   │   └── evaluator.py          # Quality and latency tests runner
│   │
│   ├── session/
│   │   └── manager.py            # Global dict storage for state
│   │
│   ├── utils/
│   │   └── helpers.py            # Reusable text processors
│   │
│   ├── api.py                    # REST API (FastAPI) — Optional
│   ├── ui.py                     # Gradio UI components and event handlers
│   └── main.py                   # Entry point for the application
│
├── data/
│   ├── uploads/                  # Uploaded documents
│   ├── vectordb/                 # FAISS vector database (by session)
│   └── cache/                    # Embeddings cache
│
├── tests/                        # Directory for application tests
│
├── .env                          # Environment variables (Do NOT push to GitHub)
├── .env.example                  # Environment template (Safe to push)
├── requirements.txt           
└── README.md
```

---

## ⚙️ Environment Variables

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `OPENROUTER_API_KEY` | — | OpenRouter API Key **(Required)** |
| `base_url` | `https://openrouter.ai/api/v1` | API base URL |
| `OPENROUTER_MODEL` | `stepfun/step-3.5-flash:free` | Model name |
| `CHUNK_SIZE` | `1000` | Text chunk size |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Number of retrieved results |
| `MAX_FILE_SIZE_MB` | `50` | Maximum allowed file size |
| `PORT` | `7860` | Gradio Port |

---

## 🌟 Features

| Feature | Details |
|---------|---------|
| 🌍 **Auto Language** | Responds in English or Arabic automatically based on your question |
| ⚡ **Instant Stream** | Answers appear token by token continuously |
| 🧮 **LaTeX** | Full support for displaying mathematical equations |
| 💬 **Chat Memory** | Remembers the last 10 messages per session |
| 📄 **File Types** | PDF · DOCX · TXT |
| 🔍 **MMR Search** | Diverse retrieval to avoid repetition |
| 📊 **Auto Evaluation** | Evaluates answer quality and saves the report |
| 🔧 **Session Restore** | Reloads the vectorstore from disk when needed |

---

## 📊 Performance Indicators

| Metric | Target | Status |
|--------|--------|--------|
| File Upload & Process | ≤ 3s | ✅ |
| Full Answer (Streaming)| ≤ 5s | ✅ |
| Retrieval (MMR) | < 1s | ✅ |
| Answer Accuracy | > 80% | Depends on the Model |

---

## 🔒 Security

- ✅ Storage is local only — data is only sent to the LLM.
- ✅ Validates file type and size before processing.
- ✅ Input sanitization.
- ✅ Auto cleanup for old sessions (every 2 hours).
- ⚠️ Do NOT upload your `.env` file to GitHub.

---

## 📚 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Gradio 4+ |
| LLM Orchestration | LangChain (LCEL) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Embedding Cache | LangChain `CacheBackedEmbeddings` |
| Vector DB | FAISS |
| LLM Provider | OpenRouter API |
| Document Loaders | PyPDF · Docx2txt · TextLoader |

---

## 🧪 Running Evaluator

```python
from app.evaluation.evaluator import SystemEvaluator

eval = SystemEvaluator()

# Test an answer
result = eval.evaluate_answer(
    question="What is the document about?",
    answer=answer,
    expected_keywords=["subject", "document"],
    latency=2.1
)

# Test latency performance
perf = eval.latency_test(ask_question, "What is the subject?", runs=5)
eval.print_summary()

# Save the report
eval.save_report("eval_report.json")
```

> 📝 Evaluation happens automatically after every real question and is saved inside `eval_report.json`.
