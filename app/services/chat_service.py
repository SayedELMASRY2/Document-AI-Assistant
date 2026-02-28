import logging
from langchain_core.messages import HumanMessage, AIMessage

from app.session.manager import get_session, update_session
from app.services.retrieval_service import get_casual_chain
from app.services.ingestion_service import restore_session
from app.utils.helpers import is_casual_message

logger = logging.getLogger(__name__)

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
        session = get_session(session_id)
        chat_history = list(session.get("chat_history", [])) if session else []
        
        casual = get_casual_chain()
        full_answer = ""
        for token in casual.stream({"question": question, "chat_history": chat_history}):
            if token:
                full_answer += token
                yield token
        
        if session:
            new_history = session.get("chat_history", [])
            new_history.append(HumanMessage(content=question))
            new_history.append(AIMessage(content=full_answer))
            update_session(session_id, {"chat_history": new_history})
            
        yield {"__sources__": []}
        return

    # ── Document RAG ─────────────────────────────────
    session = get_session(session_id)
    if not session:
        if not restore_session(session_id):
            yield "⚠️ Please upload a document first before asking questions."
            return
        session = get_session(session_id)

    chain = session["chain"]
    chat_history = list(session.get("chat_history", []))

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

        # Save history
        new_history = session.get("chat_history", [])
        new_history.append(HumanMessage(content=question))
        new_history.append(AIMessage(content=full_answer))
        
        if len(new_history) > 20:
            new_history = new_history[-20:]
            
        update_session(session_id, {"chat_history": new_history})

    except Exception as e:
        logger.exception("ask_question_stream error")
        yield f"\n\n❌ Error: {str(e)}"
