import os
import json
from typing import List, Dict, Any
from memory.sliding_memory import SlidingWindowMemory
from memory.summary_memory import SummaryMemory

class HybridMemory:
    """
    Combines SlidingWindowMemory and SummaryMemory.
    When messages exceed the sliding window capacity, they are automatically
    popped from the sliding window and merged into the SummaryMemory.
    """
    def __init__(self, window_size: int = 5):
        self.sliding_memory = SlidingWindowMemory(window_size=window_size)
        self.summary_memory = SummaryMemory()
        self.window_size = window_size
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.default_file_path = os.path.join(project_root, "memory", "chat_memory.json")
        self.load_memory()

    def add_message(self, role: str, content: str, api_key: str = None):
        """
        Adds a message. If the sliding window overflows, the oldest turn
        (user + assistant pair) is pushed to the summary memory.
        """
        # Save messages before adding
        old_messages = list(self.sliding_memory.get_messages())
        
        self.sliding_memory.add_message(role, content)
        new_messages = self.sliding_memory.get_messages()
        
        # Check if sliding window kicked out messages
        max_messages = self.window_size * 2
        if len(old_messages) >= max_messages and len(new_messages) <= max_messages:
            num_discarded = (len(old_messages) + 1) - len(new_messages)
            if num_discarded > 0:
                discarded = old_messages[:num_discarded]
                self.summary_memory.update_summary(discarded, api_key=api_key)
        self.save_memory()

    def get_context(self) -> Dict[str, Any]:
        """
        Returns the combined context containing both the summary of older chats
        and the recent chat history.
        """
        return {
            "summary": self.summary_memory.get_summary(),
            "recent_history": self.sliding_memory.get_messages()
        }

    def get_formatted_context(self) -> str:
        """
        Returns a formatted string representing the history to be fed into the model prompt.
        """
        context_str = ""
        summary = self.summary_memory.get_summary()
        if summary:
            context_str += f"--- Summary of older conversations ---\n{summary}\n\n"
            
        recent = self.sliding_memory.get_messages()
        if recent:
            context_str += "--- Recent conversation ---\n"
            for msg in recent:
                context_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
                
        return context_str

    def clear(self):
        """
        Clears both sliding window and summary memories.
        """
        self.sliding_memory.clear()
        self.summary_memory.clear()
        self.save_memory()

    def save_memory(self, file_path: str = None):
        if file_path is None:
            file_path = self.default_file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data = {
            "sliding_memory": self.sliding_memory.to_dict(),
            "summary_memory": self.summary_memory.to_dict()
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load_memory(self, file_path: str = None):
        if file_path is None:
            file_path = self.default_file_path
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "sliding_memory" in data:
                    self.sliding_memory.from_dict(data["sliding_memory"])
                if "summary_memory" in data:
                    self.summary_memory.from_dict(data["summary_memory"])
            except Exception:
                pass
