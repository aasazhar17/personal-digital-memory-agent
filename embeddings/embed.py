import os
from typing import List

class EmbeddingGenerator:
    """
    Generates semantic vector embeddings for text chunks using SentenceTransformers.
    By default, it uses the all-MiniLM-L6-v2 model, which is fast and lightweight.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """
        Lazy load the SentenceTransformer model on demand to optimize application startup.
        """
        if self._model is None:
            # We import sentence_transformers here so it's not imported unless needed
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates a vector embedding for a single string.
        """
        if not text or not text.strip():
            return []
        embedding = self.model.encode(text)
        return embedding.tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a list of strings in batch.
        """
        if not texts:
            return []
        # Filter empty texts but keep indexes if needed; here we assume valid non-empty chunks
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return []
        embeddings = self.model.encode(valid_texts)
        return embeddings.tolist()
