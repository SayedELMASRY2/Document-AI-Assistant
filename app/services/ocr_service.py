"""
OCR Service
===========
Extracts text from scanned PDFs and image files using Tesseract OCR.
Uses PyMuPDF (fitz) to render PDF pages — no Poppler required.

System requirement:
  - Tesseract OCR installed and on PATH
    Windows: https://github.com/UB-Mannheim/tesseract/wiki
"""

import io
import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed. OCR will not be available.")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed. Run: pip install pymupdf")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not installed. Image OCR will not be available.")


def ocr_available() -> bool:
    """Return True if all OCR dependencies are present."""
    return TESSERACT_AVAILABLE and PYMUPDF_AVAILABLE and PIL_AVAILABLE


def is_scanned_pdf(file_path: str, text_threshold: int = 50) -> bool:
    """
    Heuristic: a PDF is 'scanned' if it contains very little selectable text.
    Returns True if the average characters per page is below `text_threshold`.
    Uses PyMuPDF for fast text extraction.
    """
    if not PYMUPDF_AVAILABLE:
        return False
    try:
        doc = fitz.open(file_path)
        if not doc.page_count:
            return False
        total_chars = sum(len(page.get_text()) for page in doc)
        avg_chars = total_chars / doc.page_count
        doc.close()
        return avg_chars < text_threshold
    except Exception as e:
        logger.warning(f"is_scanned_pdf check failed: {e}")
        return False


def ocr_pdf(file_path: str, lang: str = "ara+eng") -> list[Document]:
    """
    Render each PDF page to an image with PyMuPDF, then run Tesseract OCR.
    No Poppler required.

    Parameters
    ----------
    file_path : str
        Absolute path to the PDF file.
    lang : str
        Tesseract language string, e.g. "ara+eng" for Arabic + English.

    Returns
    -------
    list[Document]
        One LangChain Document per page.
    """
    if not PYMUPDF_AVAILABLE:
        raise RuntimeError("PyMuPDF is not installed. Run: pip install pymupdf")
    if not TESSERACT_AVAILABLE:
        raise RuntimeError(
            "pytesseract is not installed. Run: pip install pytesseract\n"
            "Also install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki"
        )

    docs: list[Document] = []
    pdf = fitz.open(file_path)

    for page_num, page in enumerate(pdf, start=1):
        # Render page at 300 DPI for good OCR accuracy
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        text = pytesseract.image_to_string(image, lang=lang).strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": file_path, "page": page_num, "ocr": True},
            ))
        logger.debug(f"OCR page {page_num}: {len(text)} chars extracted")

    pdf.close()
    return docs


def ocr_image(file_path: str, lang: str = "ara+eng") -> list[Document]:
    """
    Run Tesseract OCR on a single image file (PNG, JPG, TIFF, BMP, etc.).

    Parameters
    ----------
    file_path : str
        Absolute path to the image file.
    lang : str
        Tesseract language string.

    Returns
    -------
    list[Document]
        A single-element list containing the extracted text as a Document.
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is not installed. Run: pip install Pillow")
    if not TESSERACT_AVAILABLE:
        raise RuntimeError(
            "pytesseract is not installed. Run: pip install pytesseract\n"
            "Also install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki"
        )

    image = Image.open(file_path)
    text = pytesseract.image_to_string(image, lang=lang).strip()

    if not text:
        return []

    return [Document(
        page_content=text,
        metadata={"source": file_path, "page": 1, "ocr": True},
    )]
