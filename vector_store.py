"""
Vector‑store helpers for standards documents.

Uses ChromaDB for storage and Gemini‑Embedding for query‑time vectors.
"""
import os
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer

import google.generativeai as genai
from google.generativeai.types import EmbedContentConfig

CHROMA_PERSIST_DIR = "./vector_db"
COLLECTION_NAME = "standards"

OFFLINE_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ONLINE_EMBED_MODEL = "gemini-embedding-001"

client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
offline_embedder = SentenceTransformer(OFFLINE_EMBED_MODEL)

def _configure_genai():
    key = os.getenv("GOOGLE_GENAI_API_KEY")
    if not key:
        raise EnvironmentError("Set GOOGLE_GENAI_API_KEY to enable online mode.")
    genai.configure(api_key=key)

def get_standards_collection():
    return client.get_or_create_collection(name=COLLECTION_NAME)

def get_offline_embedding(text: str) -> List[float]:
    return offline_embedder.encode(text, convert_to_tensor=False).tolist()

def get_online_embedding(text: str) -> List[float]:
    _configure_genai()
    response = genai.embed_content(
        model=ONLINE_EMBED_MODEL,
        content=text,
        task_type="RETRIEVAL_QUERY",
    )
    return response["embedding"]

def add_chunks_to_db(doc_name: str, chunks: List[Dict]) -> int:
    col = get_standards_collection()
    existing = col.get(where={"doc_name": doc_name}, limit=1)
    if existing and existing["ids"]:
        print(f"Document '{doc_name}' already ingested – skipping.")
        return 0

    ids = [f"{c['metadata']['doc_name']}_p{c['metadata']['page']}_c{c['metadata']['chunk']}" for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    embeddings = [get_offline_embedding(t) for t in texts]

    col.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    return len(chunks)

def query_standards(query_text: str, selected_standards: List[str], n_results: int = 5, offline_mode: bool = True) -> List[Dict]:
    col = get_standards_collection()
    embedding = get_offline_embedding(query_text) if offline_mode else get_online_embedding(query_text)
    where_filter = {"doc_name": {"$in": selected_standards}} if selected_standards else {}
    results = col.query(query_embeddings=[embedding], n_results=n_results, where=where_filter)
    return results["documents"][0] if results and results.get("documents") else []
