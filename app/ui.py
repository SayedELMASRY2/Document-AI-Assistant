import gradio as gr
import time
import shutil
import uuid
from pathlib import Path

from app.core.config import UPLOAD_DIR
from app.services.ingestion_service import ingest_document
from app.services.chat_service import ask_question_stream
from app.session.manager import clear_session_history

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
    yield history, "", "" 

    sources_text = ""
    start = time.time()
    full_answer = ""

    for chunk in ask_question_stream(msg, session_id):
        if isinstance(chunk, dict) and "__sources__" in chunk:
            
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
        else:
            full_answer += chunk
            history[-1][1] = full_answer
            yield history, "", sources_text

    yield history, "", sources_text

def clear_history(session_id: str):
    clear_session_history(session_id)
    return [], ""

# CSS
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --orange:       #f97316;
    --orange-lt:    #fb923c;
    --orange-dim:   rgba(249,115,22,.14);
    --orange-glow:  rgba(249,115,22,.35);
    --bg:           #080604;
    --surface:      #111009;
    --surface2:     #1a1610;
    --border:       rgba(249,115,22,.18);
    --text:         #faf5ee;
    --muted:        #78716c;
    --radius:       14px;
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* Ambient grid */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(249,115,22,.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(249,115,22,.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* Header */
.app-header { text-align: center; padding: 2rem 1rem 1rem; }
.app-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg,#f97316,#fbbf24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -.5px;
    margin: 0;
}
.app-sub { color: var(--muted); font-size: .88rem; margin-top: 4px; }
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--orange);
    margin-right: 6px;
    animation: pulse-dot 2s ease-in-out infinite;
    vertical-align: middle;
}
@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:.65; transform:scale(1.25); }
}

/* Chat */
.chat-wrap {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: 0 0 40px rgba(249,115,22,.06) !important;
    overflow: hidden !important;
}

/* Pill input bar */
.pill-row {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 50px !important;
    padding: 4px 4px 4px 18px !important;
    margin-top: 10px !important;
    box-shadow: 0 0 18px rgba(249,115,22,.08) !important;
}
.pill-row textarea, .pill-row input {
    background: transparent !important;
    border: none !important;
    color: var(--text) !important;
    font-size: .95rem !important;
}

/* Buttons */
.send-btn {
    background: linear-gradient(135deg,#f97316,#fb923c) !important;
    border: none !important;
    border-radius: 50px !important;
    color: #fff !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(249,115,22,.4) !important;
    transition: all .2s !important;
    min-width: 90px !important;
}
.send-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(249,115,22,.55) !important;
}
.sug-btn {
    background: var(--orange-dim) !important;
    border: 1px solid var(--border) !important;
    border-radius: 50px !important;
    color: #fcd9a0 !important;
    font-size: .78rem !important;
    font-weight: 500 !important;
    transition: all .2s !important;
}
.sug-btn:hover {
    background: rgba(249,115,22,.28) !important;
    border-color: var(--orange-lt) !important;
    transform: translateY(-1px) !important;
}
.ctrl-btn {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-size: .8rem !important;
    transition: all .2s !important;
}
.ctrl-btn:hover { border-color: var(--orange-lt)!important; color: var(--orange-lt)!important; }

/* Sidebar */
.side-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: 0 0 30px rgba(249,115,22,.05) !important;
}
.panel-label {
    font-size: .75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    color: var(--orange-lt);
    margin-bottom: 10px;
}
.upload-btn {
    background: linear-gradient(135deg,#f97316,#fb923c) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(249,115,22,.35) !important;
    transition: all .2s !important;
    width: 100% !important;
}
.upload-btn:hover { transform:translateY(-1px)!important; box-shadow:0 6px 20px rgba(249,115,22,.5)!important; }
.mono {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-size: .82rem !important;
}


.sources-acc {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

.gradio-container label, .gradio-container span,
.gradio-container p { color: var(--muted) !important; }
h1,h2,h3 { color: var(--text) !important; }
footer { display:none !important; }

/* ── User bubble → white background ── */
div[data-testid="user"] {
    background: #ffffff !important;
    border-color: rgba(255,255,255,.25) !important;
    border-radius: 18px 18px 4px 18px !important;
}
div[data-testid="user"] .prose,
div[data-testid="user"] .prose p,
div[data-testid="user"] .prose * { color: #111111 !important; }

/* ── Bot bubble → visible dark surface, bright text ── */
div[data-testid="bot"] {
    background: #1e1a14 !important;
    border: 1px solid rgba(249,115,22,.25) !important;
    border-radius: 18px 18px 18px 4px !important;
}
div[data-testid="bot"] .prose,
div[data-testid="bot"] .prose p,
div[data-testid="bot"] .prose li,
div[data-testid="bot"] .prose strong,
div[data-testid="bot"] .prose em,
div[data-testid="bot"] .prose code,
div[data-testid="bot"] .prose * { color: #f5ede0 !important; }

/* ── General chatbot text fix ── */
.chatbot .prose, .chatbot .prose p,
.chatbot .prose li, .chatbot .prose span { color: #f5ede0 !important; }

"""

def build_ui():
    theme = gr.themes.Base(
        font=gr.themes.GoogleFont("Inter"),
        primary_hue="orange",
        neutral_hue="stone",
    ).set(
        body_background_fill="#080604",
        block_background_fill="#111009",
        block_border_width="1px",
        block_border_color="rgba(249,115,22,0.18)",
        input_background_fill="#1a1610",
        button_primary_background_fill="linear-gradient(135deg,#f97316,#fb923c)",
        button_primary_text_color="#ffffff",
    )

    with gr.Blocks(title="DocuMind AI", theme=theme, css=CUSTOM_CSS) as demo:
        session_id = gr.State(str(__import__("uuid").uuid4()))
        last_msg   = gr.State("")

        # Header
        gr.HTML("""
        <div class="app-header">
            <h1 class="app-title"><span class="live-dot"></span>DocuMind AI</h1>
            <p class="app-sub">Drop a document. Ask anything. Get instant answers.</p>
        </div>
        """)

        with gr.Row(equal_height=False):

            # LEFT · Chat
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    bubble_full_width=False,
                    placeholder=(
                        "<div style='text-align:center;color:#78716c;padding:60px 0'>"
                        "<div style='font-size:2.5rem'>&#128196;</div>"
                        "<div style='margin-top:8px'>Upload a document on the right, then ask me anything.</div>"
                        "</div>"
                    ),
                    render_markdown=True,
                    show_copy_button=True,
                    elem_classes="chat-wrap",
                )

                # Pill input bar
                with gr.Row(elem_classes="pill-row"):
                    msg_input = gr.Textbox(
                        placeholder="Type your question…",
                        show_label=False,
                        scale=9,
                        container=False,
                        lines=1,
                    )
                    send_btn = gr.Button("Send ➤", scale=1, elem_classes="send-btn")

                # Suggestion chips
                with gr.Row():
                    sug1 = gr.Button("📋 Summarize",   size="sm", elem_classes="sug-btn")
                    sug2 = gr.Button("🔍 Key findings", size="sm", elem_classes="sug-btn")
                    sug3 = gr.Button("⚠️ Risks & gaps", size="sm", elem_classes="sug-btn")

                # Sources + controls
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Accordion("📚 Sources & References", open=False, elem_classes="sources-acc"):
                            sources_box = gr.Textbox(
                                show_label=False, lines=3, interactive=False,
                                placeholder="Source excerpts will appear here…",
                                elem_classes="mono",
                            )
                    with gr.Column(scale=1, min_width=150):
                        with gr.Row():
                            clear_btn = gr.Button("✕ Clear", size="sm", elem_classes="ctrl-btn")
                            regen_btn = gr.Button("🔄 Retry", size="sm", elem_classes="ctrl-btn")

            # RIGHT · Sidebar
            with gr.Column(scale=3, elem_classes="side-panel"):
                gr.HTML("<div class='panel-label'>📂 Upload Document</div>")
                file_input = gr.File(
                    label="",
                    file_types=[".pdf", ".docx", ".doc", ".txt"],
                    elem_classes="mono",
                )
                upload_btn = gr.Button("⬆  Process Document", elem_classes="upload-btn")
                upload_status = gr.Textbox(
                    label="Status", interactive=False, lines=1, elem_classes="mono",
                )

                gr.HTML("<div class='panel-label' style='margin-top:22px'>💡 Tips</div>")
                gr.HTML("""
                <div style='font-size:.8rem;color:#78716c;line-height:1.9'>
                    • Supports <b style='color:#fb923c'>PDF, DOCX, TXT</b><br>
                    • Ask in plain language<br>
                    • Sources cited automatically<br>
                    • Max file size: <b style='color:#fb923c'>50 MB</b>
                </div>
                """)

        # Events
        def chat_wrapper(msg, history, sid):
            for h, m, s in handle_chat(msg, history, sid):
                yield h, m, s, msg

        send_btn.click(
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
            sug.click(chat_wrapper, inputs=[sug, chatbot, session_id],
                      outputs=[chatbot, msg_input, sources_box, last_msg])

        upload_btn.click(handle_upload, inputs=[file_input, session_id], outputs=upload_status)

    return demo
