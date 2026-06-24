import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from vectorstore.faiss_db import FAISSDatabase

class ExpenseTool:
    """
    Manages personal expense logs, supporting persistent JSON storage,
    structured filters (e.g. after date), and semantic search via FAISS.
    """
    def __init__(self, db: FAISSDatabase, data_dir: str = None):
        self.db = db
        if data_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(project_root, "data")
        else:
            self.data_dir = data_dir
        self.file_path = os.path.join(self.data_dir, "expenses.json")
        self.expenses: List[Dict[str, Any]] = []
        self._load_expenses()

    def _load_expenses(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.expenses = json.load(f)
            except Exception:
                self.expenses = []
        else:
            self.expenses = []
            self._save_expenses()

    def _save_expenses(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.expenses, f, indent=4)

    async def add_expense(self, description: str, amount: float, date_str: str = None, category: str = "General") -> Dict[str, Any]:
        """
        Asynchronously adds an expense record, saves to disk, and indexes semantically.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._add_expense_sync, description, amount, date_str, category)

    def _add_expense_sync(self, description: str, amount: float, date_str: str = None, category: str = "General") -> Dict[str, Any]:
        if not date_str:
            date_str = datetime.today().strftime("%Y-%m-%d")
        else:
            # Clean and normalize date
            date_str = self._normalize_date(date_str)

        expense_id = len(self.expenses) + 1
        record = {
            "id": expense_id,
            "description": description,
            "amount": float(amount),
            "date": date_str,
            "category": category
        }
        self.expenses.append(record)
        self._save_expenses()

        # Semantic index entry
        text = f"Expense: {description}, Category: {category}, Amount: ₹{amount}, Date: {date_str}"
        self.db.add_texts(
            [text],
            [{
                "type": "expense",
                "expense_id": expense_id,
                "amount": float(amount),
                "date": date_str,
                "category": category,
                "description": description
            }]
        )
        return {"success": True, "expense": record}

    def _normalize_date(self, date_str: str) -> str:
        """
        Helper to normalize standard dates to YYYY-MM-DD.
        """
        date_str = date_str.strip()
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                parsed_dt = datetime.strptime(date_str, fmt)
                return parsed_dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        # If parsing fails, just return as is
        return date_str

    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Runs a semantic search specifically for expense items in the vector store.
        """
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self.db.similarity_search, query, k)
        return [r for r in results if r.get("type") == "expense"]

    async def get_expenses_after_date(self, date_str: str) -> List[Dict[str, Any]]:
        """
        Returns all expenses that occurred on or after the specified date (YYYY-MM-DD).
        """
        # Normalize date
        norm_date_str = self._normalize_date(date_str)
        try:
            target_date = datetime.strptime(norm_date_str, "%Y-%m-%d").date()
        except ValueError:
            # If target date format is invalid, parse query or return empty list
            return []

        matching = []
        for exp in self.expenses:
            try:
                exp_date_str = self._normalize_date(exp["date"])
                exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d").date()
                if exp_date >= target_date:
                    matching.append(exp)
            except Exception:
                # In case of invalid date string, skip comparison
                continue
        return matching

    async def get_all_expenses(self) -> List[Dict[str, Any]]:
        """
        Returns all stored expenses.
        """
        return self.expenses
