import os
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
from embeddings.embed import EmbeddingGenerator

class FAISSDatabase:
    """
    Manages a FAISS vector database instance.
    Handles embedding computation, indexing, persistence, and semantic searches.
    """
    def __init__(self, db_dir: str = None, embedder: EmbeddingGenerator = None):
        # Default directory relative to the project root
        if db_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_dir = os.path.join(project_root, "vectorstore", "data")
        else:
            self.db_dir = db_dir
            
        self.index_path = os.path.join(self.db_dir, "index.faiss")
        self.meta_path = os.path.join(self.db_dir, "metadata.pkl")
        self.embedder = embedder or EmbeddingGenerator()
        
        # Dimension size for all-MiniLM-L6-v2 is 384
        self.dimension = 384
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        
        self._load_or_init()

    def _load_or_init(self):
        """
        Loads the FAISS index and metadata if they exist, or initializes a new one.
        """
        import faiss
        os.makedirs(self.db_dir, exist_ok=True)
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
            except Exception as e:
                # If loading fails due to corruption or version mismatches, reset
                self._init_new_index()
        else:
            self._init_new_index()

    def _init_new_index(self):
        """
        Creates an empty IndexFlatL2 index.
        """
        import faiss
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.save()

    def save(self):
        """
        Persists the index and metadata to disk.
        """
        import faiss
        os.makedirs(self.db_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        Embeds a list of texts and inserts them into the FAISS index.
        """
        if not texts:
            return
            
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        # Ensure list lengths match
        if len(texts) != len(metadatas):
            raise ValueError("The length of texts and metadatas lists must match.")

        embeddings = self.embedder.get_embeddings(texts)
        if not embeddings:
            return
            
        embeddings_np = np.array(embeddings).astype("float32")
        
        # Add vectors to index
        self.index.add(embeddings_np)
        
        # Save corresponding text and metadata matching the exact vector index
        for text, meta in zip(texts, metadatas):
            self.metadata.append({
                "text": text,
                **meta
            })
            
        self.save()

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Runs a similarity search on the index and returns the top k results.
        Returns a list of dicts, each with 'text', other metadata, and 'score' (L2 distance).
        """
        if self.index.ntotal == 0:
            return []
            
        query_embedding = self.embedder.get_embedding(query)
        if not query_embedding:
            return []
            
        query_np = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # -1 is returned if not enough matches are found
            if idx == -1 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx].copy()
            item["score"] = float(dist)
            results.append(item)
            
        # Sort by distance ascending (IndexFlatL2 distance: 0 is closest match)
        return sorted(results, key=lambda x: x["score"])

    def clear(self):
        """
        Clears the database index and deletes all stored metadata.
        """
        self._init_new_index()
