from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/")
def read_root():
    return {"status": "online", "message": "Kairo Backend API is running!"}

# --- STUB ENDPOINTS FOR MEMBERS A & B ---

@app.post("/api/auth/verify-face")
async def verify_face(file: UploadFile = File(...)):
    # Member A will plug in DeepFace logic here
    return {"status": "success", "user": "verified_user", "confidence": 0.98}

@app.post("/api/rag/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Member B will plug in PyMuPDF + ChromaDB logic here
    return {"status": "success", "filename": file.filename, "chunks_indexed": 12}

@app.post("/api/chat")
async def chat_query(prompt: str = Form(...)):
    # Member B will plug in LLM + RAG response generation here
    return {"status": "success", "response": f"Kairo Echo: {prompt}"}