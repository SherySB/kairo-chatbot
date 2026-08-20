import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Member B's RAG Imports
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

# Initialize Member B's RAG Service
rag = RAGService()


@app.get("/")
def read_root():
    return {"status": "online", "message": "Kairo Backend API is running!"}

# --- STUB ENDPOINT FOR MEMBER A (Auth) ---


@app.post("/api/auth/verify-face")
async def verify_face(file: UploadFile = File(...)):
    # Member A will replace this with real DeepFace verification
    return {"status": "success", "user": "verified_user", "confidence": 0.98}

# --- REAL RAG ENDPOINTS (MEMBER B INTEGRATION) ---


@app.post("/api/rag/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)

        # Save PDF locally temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse and chunk PDF using Member B's services
        text = extract_text_from_pdf(file_path)
        if not text.strip():
            raise HTTPException(
                status_code=400, detail="Failed to extract text from PDF.")

        chunks = chunk_text(text, chunk_size=500, overlap=50)
        rag.index_chunks(chunks, doc_id=file.filename)

        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_query(prompt: str = Form(...)):
    try:
        # Perform retrieval and query Groq LLM
        answer = rag.query(prompt)
        return {"status": "success", "response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
