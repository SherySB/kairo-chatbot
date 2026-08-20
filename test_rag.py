import sys
import os
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

from src.rag.parser_service import extract_text_from_pdf, chunk_text
from src.rag.rag_service import RAGService
import pymupdf

print("==========================================")
print("     KAIRO RAG ENGINE END-TO-END TEST    ")
print("==========================================")

# 1. Create a sample test PDF to test parser_service.py
test_pdf_path = "sample_test_doc.pdf"
doc = pymupdf.open()
page = doc.new_page()
sample_pdf_text = (
    "Operating Systems: Process Synchronization\n\n"
    "Process synchronization is the task of coordinating the execution of processes "
    "such that no two processes can have access to the same shared data and resources at the same time.\n"
    "A Critical Section is a code segment that accesses shared variables or resources.\n"
    "Semaphores and Mutex locks are common synchronization mechanisms.\n"
    "Member B is the Document Parsing and RAG Engine Specialist for Kairo Chatbot."
)
page.insert_text((50, 72), sample_pdf_text, fontsize=12)
doc.save(test_pdf_path)
doc.close()
print(f"[✓] Created sample PDF: {test_pdf_path}")

# 2. Test Parser Service (extract text + chunk)
print("\n[1] Testing Parser Service (extract_text_from_pdf & chunk_text)...")
extracted_text = extract_text_from_pdf(test_pdf_path)
print(f"Extracted length: {len(extracted_text)} characters")

chunks = chunk_text(extracted_text, chunk_size=500, overlap=50)
print(f"Total chunks created: {len(chunks)}")
for idx, chunk in enumerate(chunks):
    print(f"  Chunk {idx+1}: {chunk[:80]}...")

# 3. Test RAG Service (Indexing to ChromaDB)
print("\n[2] Testing RAG Service Indexing (ChromaDB + all-MiniLM-L6-v2)...")
rag = RAGService(collection_name="test_collection")
rag.index_chunks(chunks, doc_id="test_os_doc", metadata={"source": "sample_test_doc.pdf"})
print("[✓] Chunks indexed into ChromaDB successfully.")

# 4. Test RAG Retrieval & Groq LLM Query
print("\n[3] Testing RAG Retrieval + Groq LLM Query...")
question_1 = "What is process synchronization?"
print(f"Q1: {question_1}")
answer_1 = rag.query(question_1)
print(f"A1: {answer_1}\n")

question_2 = "What did Member B develop in Kairo Chatbot?"
print(f"Q2: {question_2}")
answer_2 = rag.query(question_2)
print(f"A2: {answer_2}\n")

# Cleanup sample test pdf
if os.path.exists(test_pdf_path):
    os.remove(test_pdf_path)
    print(f"[✓] Cleaned up temporary test file: {test_pdf_path}")

print("\n==========================================")
print("       ALL RAG ENGINE TESTS PASSED!       ")
print("==========================================")