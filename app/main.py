import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent.parent / ".env")

from app.ui import build_ui

if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    print("🚀 Starting smart contract assistant ")
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=True,
    )