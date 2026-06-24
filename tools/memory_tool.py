import os
import json
from typing import Dict, Any, List

class MemoryTool:
    """
    Manages structured user profile facts (name, city, college, pet name, etc.).
    Saves profile to data/user_profile.json and indexes facts to FAISS.
    """
    def __init__(self, db, data_dir: str = None):
        self.db = db
        if data_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(project_root, "data")
        else:
            self.data_dir = data_dir
        self.file_path = os.path.join(self.data_dir, "user_profile.json")
        self.profile: Dict[str, str] = {
            "name": "",
            "city": "",
            "hometown": "",
            "college": "",
            "profession": "",
            "birthday": "",
            "pet name": "",
            "mother name": "",
            "father name": "",
            "favourite food": "",
            "favourite movie": "",
            "hobbies": ""
        }
        self._load_profile()

    def _load_profile(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Keep keys consistent
                for k in self.profile:
                    if k in data:
                        self.profile[k] = data[k]
                    # Also support spelling variant of favorite food
                    elif k == "favourite food" and "favorite food" in data:
                        self.profile[k] = data["favorite food"]
            except Exception:
                pass
        else:
            self._save_profile()

    def _save_profile(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.profile, f, indent=4)

    def store_fact(self, key: str, value: str) -> Dict[str, Any]:
        """
        Stores a fact key-value pair and indexes it in FAISS.
        """
        clean_key = key.strip().lower()
        clean_val = value.strip()
        
        # Standardize key names
        if clean_key in ["favorite food", "favourite food"]:
            clean_key = "favourite food"
        elif clean_key in ["favorite movie", "favourite movie"]:
            clean_key = "favourite movie"
        elif clean_key in ["pet", "dog", "cat", "pet name", "dog name", "cat name"]:
            clean_key = "pet name"
        elif clean_key in ["location", "town", "city"]:
            clean_key = "city"
        elif clean_key in ["hometown", "home town"]:
            clean_key = "hometown"
        elif clean_key in ["mother", "mother name", "mother's name"]:
            clean_key = "mother name"
        elif clean_key in ["father", "father name", "father's name"]:
            clean_key = "father name"
        elif clean_key in ["hobby", "hobbies"]:
            clean_key = "hobbies"
            
        self.profile[clean_key] = clean_val
        self._save_profile()

        # Semantic index entry
        text = f"User Profile Memory Fact: my {clean_key} is {clean_val}"
        self.db.add_texts(
            [text],
            [{
                "type": "profile_fact",
                "key": clean_key,
                "value": clean_val
            }]
        )
        return {"success": True, "key": clean_key, "value": clean_val}

    def retrieve_fact(self, key: str) -> str:
        """
        Retrieves a fact value by key.
        """
        clean_key = key.strip().lower()
        if clean_key in ["favorite food", "favourite food"]:
            clean_key = "favourite food"
        elif clean_key in ["favorite movie", "favourite movie"]:
            clean_key = "favourite movie"
        elif clean_key in ["pet", "dog", "cat", "pet name", "dog name", "cat name"]:
            clean_key = "pet name"
        elif clean_key in ["location", "town", "city"]:
            clean_key = "city"
        elif clean_key in ["hometown", "home town"]:
            clean_key = "hometown"
        elif clean_key in ["mother", "mother name", "mother's name"]:
            clean_key = "mother name"
        elif clean_key in ["father", "father name", "father's name"]:
            clean_key = "father name"
        elif clean_key in ["hobby", "hobbies"]:
            clean_key = "hobbies"
        return self.profile.get(clean_key, "")

    def search_memory(self, query: str) -> List[Dict[str, str]]:
        """
        Searches profile keys and returns matching key-value dicts.
        """
        query_clean = query.lower()
        matches = []
        for k, v in self.profile.items():
            if v and (k in query_clean or k.replace("favourite", "favorite") in query_clean):
                matches.append({"key": k, "value": v})
        return matches

    def get_all_facts(self) -> Dict[str, str]:
        """
        Returns all non-empty profile facts.
        """
        return {k: v for k, v in self.profile.items() if v}

    def get_formatted_context(self) -> str:
        """
        Formats all non-empty profile facts as system context.
        """
        non_empty = self.get_all_facts()
        if not non_empty:
            return ""
        lines = []
        for k, v in non_empty.items():
            lines.append(f"- User's {k}: {v}")
        return "User Profile Facts:\n" + "\n".join(lines)

    def clear(self):
        """
        Resets profile facts to empty values.
        """
        for k in self.profile:
            self.profile[k] = ""
        self._save_profile()
