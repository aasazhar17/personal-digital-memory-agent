import os
import json
from typing import Dict, Any, List

class ProfileTool:
    """
    Manages personal user facts (e.g. name, favourite food).
    Integrates JSON persistence and FAISS vector database indexing.
    """
    def __init__(self, db, data_dir: str = None):
        self.db = db
        if data_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(project_root, "data")
        else:
            self.data_dir = data_dir
        self.file_path = os.path.join(self.data_dir, "profile.json")
        self.profile: Dict[str, str] = {}
        self._load_profile()

    def _load_profile(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.profile = json.load(f)
            except Exception:
                self.profile = {}
        else:
            self.profile = {}
            self._save_profile()

    def _save_profile(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.profile, f, indent=4)

    def add_fact(self, key: str, value: str) -> Dict[str, Any]:
        """
        Adds or updates a personal fact. Indexes it semantically in FAISS.
        """
        # Normalize keys: strip, lower, clean
        clean_key = key.strip().lower()
        clean_val = value.strip()
        
        # Save to dict
        self.profile[clean_key] = clean_val
        self._save_profile()

        # Semantic index entry
        text = f"User Profile Fact: my {clean_key} is {clean_val}"
        self.db.add_texts(
            [text],
            [{
                "type": "profile",
                "key": clean_key,
                "value": clean_val
            }]
        )
        return {"success": True, "key": clean_key, "value": clean_val}

    def get_fact(self, key: str) -> str:
        """
        Retrieves a fact value. Case-insensitive key match.
        """
        return self.profile.get(key.strip().lower(), None)

    def get_all_facts(self) -> Dict[str, str]:
        """
        Returns all saved user facts.
        """
        return self.profile

    def get_formatted_context(self) -> str:
        """
        Returns facts formatted as a context string for LLM or system prompt.
        """
        if not self.profile:
            return ""
        lines = []
        for k, v in self.profile.items():
            lines.append(f"- User's {k}: {v}")
        return "User Profile Facts:\n" + "\n".join(lines)

    def clear(self):
        """
        Clears all profile facts.
        """
        self.profile = {}
        self._save_profile()
