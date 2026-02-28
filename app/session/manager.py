from typing import Dict, Optional, Any
from datetime import datetime
from app.core.locks import sessions_lock

# In-memory session store
_sessions: Dict[str, Dict[str, Any]] = {}

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a session, returning None if it doesn't exist."""
    with sessions_lock:
        return _sessions.get(session_id)

def set_session(session_id: str, data: Dict[str, Any]) -> None:
    """Create or overwrite a session with new data."""
    with sessions_lock:
        _sessions[session_id] = {
            **data,
            "last_used": datetime.now(),
        }

def update_session(session_id: str, keys_to_update: Dict[str, Any]) -> None:
    """Update specific keys in an existing session."""
    with sessions_lock:
        if session_id in _sessions:
            _sessions[session_id].update(keys_to_update)
            _sessions[session_id]["last_used"] = datetime.now()

def clear_session_history(session_id: str) -> None:
    """Clear the chat history of a session."""
    with sessions_lock:
        if session_id in _sessions:
            _sessions[session_id]["chat_history"] = []
