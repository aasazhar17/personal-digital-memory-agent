import re
import asyncio
from typing import List, Dict, Any
from embeddings.embed import EmbeddingGenerator

class AgentRouter:
    """
    Orchestrates routing of user queries to the correct tools.
    Supports LLM-based routing using Gemini, and falls back to local
    semantic + keyword similarity matching.
    """
    def __init__(self, embedder: EmbeddingGenerator = None):
        self.embedder = embedder or EmbeddingGenerator()
        # Semantic mapping descriptions for local matching
        self.tool_descriptions = {
            "pdf_tool": [
                "search uploaded PDF documents and statements",
                "MRI reports, medical scans, clinic prescriptions, lab results",
                "bank statements, financial statements, PDF document search",
                "electricity bills, utility bills, official documents"
            ],
            "expense_tool": [
                "log expenses, spendings, payments, prices",
                "how much money was spent, expenditures, costs",
                "Goa trip expense, flights price, hotel bill, travel cost",
                "electricity bill amount, monthly spendings, financial ledger"
            ],
            "notes_tool": [
                "read notes, search notes, view reminders, personal memos",
                "Goa trip budget, Goa trip itinerary details, travel notes",
                "EMI due date, EMI 10th of every month, loan dates",
                "general text logs, personal memories"
            ],
            "calculator_tool": [
                "calculate mathematical equations, sum numbers, arithmetic",
                "add values, subtract costs, multiply rates, divide bills",
                "perform calculations, mathematical totals"
            ],
            "memory_tool": [
                "personal facts, name, favorite food, pet name, city, college",
                "what is my name, what is my pet name, where do I live",
                "who am I, tell me about myself, what do you know about me",
                "my favorite color, save my profile, user details"
            ]
        }
        self.tool_keys = list(self.tool_descriptions.keys())
        self._embedded_descriptions = {}

    def _get_embedded_descriptions(self) -> Dict[str, List[List[float]]]:
        """
        Lazy embedding of tool description keys for semantic search logic.
        """
        if not self._embedded_descriptions:
            for tool, desc_list in self.tool_descriptions.items():
                self._embedded_descriptions[tool] = self.embedder.get_embeddings(desc_list)
        return self._embedded_descriptions

    async def route(self, query: str, api_key: str = None) -> List[str]:
        """
        Routes query to appropriate tools. Uses Gemini if api_key is provided,
        otherwise falls back to rule-based + semantic vector similarity routing.
        """
        if api_key:
            try:
                import google.generativeai as genai
                import json
                
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                prompt = (
                    f"You are the routing system for an AI Personal Assistant.\n"
                    f"Decide which tools from the list below are necessary to answer the user query.\n\n"
                    f"Available Tools:\n"
                    f"1. pdf_tool: Search uploaded documents like MRI reports, medical statements, bills, bank PDFs.\n"
                    f"2. expense_tool: Check logged expenses, list costs, Goa trip cost, flights, electricity bill amount.\n"
                    f"3. notes_tool: Access notes, travel dates, Goa trip budget, EMI due dates, textual reminders.\n"
                    f"4. calculator_tool: Solve math equations, sum up values, calculate differences, evaluate arithmetic expressions.\n"
                    f"5. memory_tool: Save, update, and search personal profile facts (name, city, college, profession, pet name, birthday, favourite food).\n\n"
                    f"User Query: \"{query}\"\n\n"
                    f"Response MUST be a JSON list containing only tool names. Select multiple tools if the user is asking a complex question requiring multiple tools. "
                    f"Example response format: [\"notes_tool\", \"expense_tool\"]\n"
                    f"Return ONLY valid JSON and nothing else."
                )
                
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None, lambda: model.generate_content(prompt)
                )
                
                resp_text = response.text.strip()
                if "```" in resp_text:
                    resp_text = resp_text.replace("```json", "").replace("```", "").strip()
                
                selected = json.loads(resp_text)
                if isinstance(selected, list):
                    # Filter only valid tool names
                    valid_selections = [t for t in selected if t in self.tool_keys]
                    if valid_selections:
                        return valid_selections
            except Exception as e:
                # If API fail, proceed to local matching
                pass

        return self.route_local(query)

    def route_local(self, query: str) -> List[str]:
        """
        Performs local fallback routing using regex heuristics and vector cosine similarity.
        """
        import numpy as np
        query_clean = query.lower()
        selected_tools = set()

        # 1. Rule-based triggers (Regex / Substring checks)
        # Math & arithmetic
        math_symbols = ["+", "-", "*", "/", "%", "="]
        if any(sym in query_clean for sym in math_symbols) or re.search(r'\d+\s*(plus|minus|times|divided by|x|\+)\s*\d+', query_clean):
            selected_tools.add("calculator_tool")

        # Memory / Profile Facts
        memory_keywords = ["what is my", "what's my", "what do you know about me", "tell me about myself", "my name", "my city", "my college", "my favourite food", "my favorite food", "my pet", "my birthday", "who am i", "my profile", "do you know my", "tell me my"]
        if any(w in query_clean for w in memory_keywords) or re.search(r"\bmy\s+(favourite\s+\w+|favorite\s+\w+|name|age|location|city|country|hobby|color|colour|job|profession|food|pet|book|movie|birthday|college)\s+is\s+(.+)", query_clean):
            selected_tools.add("memory_tool")

        # Documents & Reports (PDFs)
        pdf_keywords = ["pdf", "document", "ticket", "invoice", "report", "bill", "receipt", "cost", "amount", "booking", "transaction", "pnr", "railway", "medical", "hospital", "travel", "mri", "statement", "prescription", "scan", "sifa", "shifa", "mummy"]
        if any(w in query_clean for w in pdf_keywords):
            selected_tools.add("pdf_tool")

        # Expenses
        expense_keywords = ["spend", "spent", "expense", "expenditure", "budget", "paid", "buy", "bought", "purchase", "ledger", "cost", "amount", "price"]
        if any(w in query_clean for w in expense_keywords):
            # Prioritize pdf_tool for document-specific keywords, and only route to expense_tool if we don't have document terms, OR if the query specifically mentions trip/spend/ledger
            is_doc_only = any(dw in query_clean for dw in ["ticket", "pnr", "railway", "mri", "prescription", "scan", "hospital", "medical"])
            if not is_doc_only or any(ew in query_clean for ew in ["budget", "ledger", "spend", "spent", "expense"]):
                selected_tools.add("expense_tool")

        # Notes & Reminders
        notes_keywords = ["note", "memo", "emi", "due", "remind", "trip budget", "trip date", "date", "schedule"]
        if any(w in query_clean for w in notes_keywords):
            selected_tools.add("notes_tool")

        # Memory / Profile Facts
        memory_keywords = ["what is my", "what's my", "what do you know about me", "tell me about myself", "my name", "my city", "my college", "my favourite food", "my favorite food", "my pet", "my birthday", "who am i", "my profile"]
        if any(w in query_clean for w in memory_keywords) or re.search(r"\bmy\s+(favourite\s+\w+|favorite\s+\w+|name|age|location|city|country|hobby|color|colour|job|profession|food|pet|book|movie|birthday|college)\s+is\s+(.+)", query_clean):
            selected_tools.add("memory_tool")

        # 2. Vector Cosine Similarity Semantic Checks
        try:
            query_emb = np.array(self.embedder.get_embedding(query))
            if len(query_emb) > 0:
                embedded_desc = self._get_embedded_descriptions()
                semantic_scores = {}
                
                for tool, desc_embs in embedded_desc.items():
                    max_sim = 0.0
                    for emb in desc_embs:
                        emb_np = np.array(emb)
                        # Cosine similarity
                        dot_product = np.dot(query_emb, emb_np)
                        norm_q = np.linalg.norm(query_emb)
                        norm_e = np.linalg.norm(emb_np)
                        if norm_q > 0 and norm_e > 0:
                            sim = dot_product / (norm_q * norm_e)
                            max_sim = max(max_sim, sim)
                    semantic_scores[tool] = max_sim

                # Threshold mapping
                for tool, score in semantic_scores.items():
                    if score > 0.42:
                        selected_tools.add(tool)

                # Fallback to absolute highest scoring tool if nothing triggered
                if not selected_tools:
                    top_tool = max(semantic_scores, key=semantic_scores.get)
                    selected_tools.add(top_tool)
        except Exception:
            # In case sentence-transformers load fails or numpy fails, default if empty
            if not selected_tools:
                selected_tools.add("notes_tool")

        return list(selected_tools)
