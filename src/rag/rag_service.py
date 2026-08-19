import os
from typing import List
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

class RAGService:
    def __init__(self, collection_name: str = "pdf_rag"):
        # 1. In-memory ChromaDB client
        self.chroma_client = chromadb.Client()
        
        # 2. Embedding Model
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, 
            embedding_function=self.emb_fn
        )
        
        # 3. Groq LLM Client
        api_key = os.getenv("GROQ_API_KEY", "")
        self.llm_client = Groq(api_key=api_key) if api_key else None

    def index_chunks(self, chunks: List[str], doc_id: str = "doc"):
        """Chunks ko ChromaDB vector database mein store karta hai."""
        if not chunks:
            return
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            ids=ids
        )

    def retrieve_context(self, query: str, n_results: int = 3) -> str:
        """Top-3 matching context dhoondta hai."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        matched_docs = results.get("documents", [[]])[0]
        return "\n\n".join(matched_docs)

    def generate_response(self, query: str, context: str) -> str:
        """LLM se context-aware answer generate karwata hai."""
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
            model="llama-3.1-8b-instant",
        )
        return response.choices[0].message.content

    def query(self, user_query: str) -> str:
        """Complete RAG query execution."""
        context = self.retrieve_context(user_query)
        if not context.strip():
            return "No relevant document context found."
        return self.generate_response(user_query, context)