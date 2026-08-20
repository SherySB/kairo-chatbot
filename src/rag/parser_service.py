import pymupdf  # PyMuPDF
from typing import List

def extract_text_from_pdf(pdf_source) -> str:
    """PDF file path, bytes, ya stream se saara text extract karta hai."""
    try:
        if isinstance(pdf_source, (bytes, bytearray)):
            doc = pymupdf.open(stream=pdf_source, filetype="pdf")
        elif hasattr(pdf_source, "read"):
            data = pdf_source.read()
            doc = pymupdf.open(stream=data, filetype="pdf")
        else:
            doc = pymupdf.open(str(pdf_source))

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