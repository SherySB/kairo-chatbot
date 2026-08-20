import os
import uuid
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class RAGService:
    def __init__(self, collection_name: str = "pdf_rag", persist_dir: str = "./chroma_db", model_name: str = "openai/gpt-oss-20b"):
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.emb_fn
        )
        self.model_name = os.getenv("GROQ_MODEL", model_name)
        api_key = os.getenv("GROQ_API_KEY", "")
        self.llm_client = Groq(api_key=api_key) if api_key else None

    def index_chunks(self, chunks: List[str], doc_id: str = "doc", metadata: Optional[dict] = None):
        if not chunks:
            return
        ids = [f"{doc_id}_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))]
        metadatas = [metadata or {"source": doc_id} for _ in chunks]

        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

    def retrieve_context(self, query: str, n_results: int = 3) -> str:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        matched_docs = results.get("documents", [[]])[0]
        return "\n\n".join(matched_docs)

    def generate_response(self, query: str, context: str) -> str:
        if not self.llm_client:
            return "Error: GROQ_API_KEY environment variable set nahi hai."

        prompt = f"""You are a helpful assistant. Use the following context to answer the user question accurately.
If the answer cannot be found in the context, strictly say: "I do not have enough information from the document to answer this."

Context:
{context}

Question:
{query}

Answer:"""

        response = self.llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
        )
        return response.choices[0].message.content

    def query(self, user_query: str) -> str:
        context = self.retrieve_context(user_query)
        if not context.strip():
            return "No relevant document context found."
        return self.generate_response(user_query, context)