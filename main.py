import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.rag.parser_service import extract_text_from_pdf, chunk_text
from src.rag.rag_service import RAGService

app = FastAPI(
    title="Kairo AI Assistant API",
    description="Backend API endpoints for authentication, RAG search, and voice handling.",
    version="1.0.0"
)

# Enable CORS for local Streamlit communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Service
rag_service = RAGService()

@app.get("/")
def read_root():
    return {"status": "online", "message": "Kairo Backend API is running!"}

# --- AUTHENTICATION ENDPOINT ---

@app.post("/api/auth/verify-face")
async def verify_face(file: UploadFile = File(...)):
    # Member A will plug in DeepFace logic here
    return {"status": "success", "user": "verified_user", "confidence": 0.98}

# --- RAG & CHAT ENDPOINTS (MEMBER B) ---

@app.post("/api/rag/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        # Extract text using PyMuPDF
        text = extract_text_from_pdf(tmp_path)

        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in the uploaded PDF.")

        # Chunk text into 500-character pieces with 50-character overlap
        chunks = chunk_text(text, chunk_size=500, overlap=50)

        # Index chunks in ChromaDB vector database
        rag_service.index_chunks(chunks, doc_id=file.filename, metadata={"source": file.filename})

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": len(chunks),
            "message": f"Successfully indexed {len(chunks)} chunks into ChromaDB."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_query(prompt: str = Form(...)):
    try:
        response = rag_service.query(prompt)
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))