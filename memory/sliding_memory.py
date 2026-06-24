from typing import List, Dict

class SlidingWindowMemory:
    """
    Maintains a sliding window of the most recent N conversation turns.
    Each turn is stored as a dictionary: {"role": "user" | "assistant", "content": "..."}
    """
    def __init__(self, window_size: int = 5):
        # window_size is the number of complete user-assistant dialogue pairs.
        # So maximum messages stored is window_size * 2.
        self.window_size = window_size
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """
        Adds a message to the memory and clips if it exceeds the window size.
        """
        self.messages.append({"role": role, "content": content})
        # Slide window if necessary
        max_messages = self.window_size * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Returns all messages currently held in the sliding window.
        """
        return self.messages

    def clear(self):
        """
        Clears the sliding window memory.
        """
        self.messages = []

    def to_dict(self) -> dict:
        return {
            "window_size": self.window_size,
            "messages": self.messages
        }

    def from_dict(self, data: dict):
        self.window_size = data.get("window_size", self.window_size)
        self.messages = data.get("messages", [])
