import os
import chromadb
from chromadb.config import Settings

def _clean_host(raw: str) -> str:
    if not raw:
        return raw
    return raw.replace("http://", "").replace("https://", "").rstrip("/")

class VectorORM:
    def __init__(self):
        # Expect these to be set in Railway variables:
        # CHROMA_HOST=chromatushar.railway.internal
        # CHROMA_PORT=8000
        raw_host = os.getenv("CHROMA_HOST", "chromatushar.railway.internal")
        CHROMA_HOST = _clean_host(raw_host)
        CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

        # Use the v2 REST client settings — DO NOT pass ssl_enabled (invalid)
        settings = Settings(
            chroma_api_impl="rest",
            chroma_server_host=CHROMA_HOST,
            chroma_server_http_port=CHROMA_PORT,
        )

        self.client = chromadb.Client(settings)

        self.predefined = "predefined_context"
        self.user_history = "user_history"

        self._ensure_collection(self.predefined)
        self._ensure_collection(self.user_history)

    def _ensure_collection(self, name):
        try:
            self.client.get_collection(name)
        except Exception:
            self.client.create_collection(name=name)

    def insert(self, collection, text, embedding, metadata):
        col = self.client.get_collection(collection)
        col.add(
            ids=[metadata["id"]],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    def search(self, collection, embedding, limit=4):
        col = self.client.get_collection(collection)
        return col.query(query_embeddings=[embedding], n_results=limit)
