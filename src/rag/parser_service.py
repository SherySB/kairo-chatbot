import fitz  # PyMuPDF
from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF file se saara text extract karta hai."""
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        for page in doc:
            text = page.get_text()
            if text:
                full_text.append(text)
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Text ko overlapping chunks mein split karta hai."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks