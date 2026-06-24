# 🧠 Personal Digital Memory Agent

An Agentic AI + RAG based personal assistant that stores and retrieves documents, expenses, notes, and user memories using semantic search.

## 🚀 Features

- PDF document ingestion and retrieval
- Expense management
- Personal notes management
- Memory facts (Name, City, Favourite Food, etc.)
- Hybrid Memory (Sliding Window + Summary Memory)
- FAISS Vector Database
- Semantic Search using Sentence Transformers
- Agent Router for dynamic tool selection
- Streamlit UI

## 🛠 Technologies Used

- Python
- Streamlit
- FAISS
- Sentence Transformers
- PyPDF
- Asyncio
- JSON
- Agentic AI
- RAG (Retrieval Augmented Generation)

## 📂 Project Structure

```
agent/
memory/
tools/
vectorstore/
documents/
data/
ui/
app.py
requirements.txt
```

## ⚙️ How It Works

User Query
↓
Agent Router
↓
Tool Selection
↓
FAISS Vector Database
↓
Relevant Chunks Retrieval
↓
Hybrid Memory
↓
LLM / Synthesizer
↓
Final Answer

## 🧠 Memory System

### Short-Term Memory
- Sliding Window Memory

### Long-Term Memory
- Summary Memory

### Profile Memory
Stores:
- Name
- City
- College
- Favourite Food
- Pet Name

## 📌 Example Queries

- What was my Goa trip budget?
- How much was my electricity bill?
- What is my favourite food?
- When is my EMI due?
- Summarize my MRI report.

## 📈 Future Scope

- Gmail Integration
- Calendar Integration
- WhatsApp Integration
- Gemini/OpenAI APIs
- Voice Assistant
- OCR Support
- Multi-Agent Architecture

## 👨‍💻 Author

Azhar Ali  
B.Tech CSE (AI & ML)  
K.R. Mangalam University
