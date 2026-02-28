from pathlib import Path

UPLOAD_DIR   = Path("data/uploads")
VECTORDB_DIR = Path("data/vectordb")
CACHE_DIR    = Path("data/cache")
MAX_FILE_MB  = 50
SESSION_TIMEOUT_HOURS = 2

for d in [UPLOAD_DIR, VECTORDB_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
