from typing import List, Dict

class SummaryMemory:
    """
    Maintains a running, condensed summary of older conversational turns.
    Integrates new messages that have slid out of the active window.
    """
    def __init__(self):
        self.summary: str = ""

    def update_summary(self, messages_to_add: List[Dict[str, str]], api_key: str = None):
        """
        Updates the running summary with new dialogue lines.
        If a Gemini API key is provided, uses gemini-1.5-flash to write a cohesive, 
        fact-dense summary. Otherwise, falls back to a structural text summarizer.
        """
        if not messages_to_add:
            return

        dialogue_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages_to_add])
        
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                prompt = (
                    f"You are a memory compressor for a personal assistant. "
                    f"Here is the existing running summary of the conversation:\n"
                    f"\"\"\"\n{self.summary}\n\"\"\"\n\n"
                    f"And here are the new conversational turns to append and integrate:\n"
                    f"\"\"\"\n{dialogue_text}\n\"\"\"\n\n"
                    f"Provide an updated, concise running summary that integrates these details. "
                    f"Do not lose key specific facts like dates, numbers, prices, or documents mentioned."
                )
                response = model.generate_content(prompt)
                self.summary = response.text.strip()
                return
            except Exception as e:
                # Fallback to local logic if API call fails
                pass

        # Offline / Heuristic Fallback
        lines = []
        for msg in messages_to_add:
            snippet = msg["content"]
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            lines.append(f"- {msg['role'].capitalize()}: {snippet}")
        
        added_summary = "\n".join(lines)
        if self.summary:
            self.summary = f"{self.summary}\n{added_summary}"
        else:
            self.summary = f"Summary of earlier chat history:\n{added_summary}"

        # Limit local summary length to avoid infinite growth
        summary_lines = self.summary.split("\n")
        if len(summary_lines) > 30:
            # Keep header and last 20 lines
            self.summary = summary_lines[0] + "\n" + "\n".join(summary_lines[-20:])

    def get_summary(self) -> str:
        """
        Returns the running summary text.
        """
        return self.summary

    def clear(self):
        """
        Clears the summary.
        """
        self.summary = ""

    def to_dict(self) -> dict:
        return {
            "summary": self.summary
        }

    def from_dict(self, data: dict):
        self.summary = data.get("summary", "")
