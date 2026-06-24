import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from vectorstore.faiss_db import FAISSDatabase

class NotesTool:
    """
    Manages personal text notes. Integrates JSON persistence and FAISS vector search.
    """
    def __init__(self, db: FAISSDatabase, data_dir: str = None):
        self.db = db
        if data_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(project_root, "data")
        else:
            self.data_dir = data_dir
        self.file_path = os.path.join(self.data_dir, "notes.json")
        self.notes: List[Dict[str, Any]] = []
        self._load_notes()

    def _load_notes(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.notes = json.load(f)
            except Exception:
                self.notes = []
        else:
            self.notes = []
            self._save_notes()

    def _save_notes(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.notes, f, indent=4)

    async def add_note(self, content: str, title: str = None) -> Dict[str, Any]:
        """
        Asynchronously adds a note and indexes it in FAISS.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._add_note_sync, content, title)

    def _add_note_sync(self, content: str, title: str = None) -> Dict[str, Any]:
        if not title:
            words = content.strip().split()
            title = " ".join(words[:5]) + ("..." if len(words) > 5 else "")

        note_id = len(self.notes) + 1
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "id": note_id,
            "title": title,
            "content": content,
            "created_at": created_at
        }
        self.notes.append(record)
        self._save_notes()

        # Semantic index entry
        text = f"Note Title: {title}\nNote Content: {content}\nDate: {created_at}"
        self.db.add_texts(
            [text],
            [{
                "type": "note",
                "note_id": note_id,
                "title": title,
                "content": content,
                "created_at": created_at
            }]
        )
        return {"success": True, "note": record}

    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Runs a semantic search specifically for notes in the vector database.
        """
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self.db.similarity_search, query, k)
        return [r for r in results if r.get("type") == "note"]

    async def get_all_notes(self) -> List[Dict[str, Any]]:
        """
        Returns all saved notes.
        """
        return self.notes
