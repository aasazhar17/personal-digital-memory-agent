import os
import json
import asyncio
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any

from vectorstore.faiss_db import FAISSDatabase
from tools.pdf_tool import PDFTool
from tools.expense_tool import ExpenseTool
from tools.notes_tool import NotesTool
from tools.calculator_tool import CalculatorTool
from tools.memory_tool import MemoryTool
from agent.router import AgentRouter
from agent.planner import AgentPlanner
from memory.hybrid_memory import HybridMemory

# Helper to load and save Streamlit chat history to disk
def load_chat_history() -> List[Dict[str, Any]]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    history_path = os.path.join(project_root, "data", "chat_history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_chat_history(history: List[Dict[str, Any]]):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    history_path = os.path.join(project_root, "data", "chat_history.json")
    try:
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
    except Exception:
        pass

# Helper to create a valid minimal PDF dynamically
def create_minimal_pdf(file_path: str, text: str):
    """
    Generates a valid, minimal 1-page PDF file with dynamic byte offsets
    so that pypdf can read it without errors.
    """
    body = []
    offsets = []
    header = b"%PDF-1.4\n"
    body.append(header)
    
    def add_obj(data: bytes) -> str:
        obj_num = len(offsets) + 1
        offset = sum(len(x) for x in body)
        offsets.append(offset)
        obj_header = f"{obj_num} 0 obj\n".encode('ascii')
        body.append(obj_header + data + b"\nendobj\n")
        return f"{obj_num} 0 R"

    # Catalog Object
    add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")
    # Pages tree
    add_obj(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    # Page definition
    add_obj(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")
    # Contents stream
    stream_content = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET".encode('latin-1')
    stream_len = len(stream_content)
    add_obj(f"<< /Length {stream_len} >>\nstream\n".encode('ascii') + stream_content + b"\nendstream")
    # Font definition
    add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    
    # Cross-reference table offset
    xref_offset = sum(len(x) for x in body)
    xref_header = f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f\n".encode('ascii')
    body.append(xref_header)
    
    for off in offsets:
        body.append(f"{off:010d} 00000 n\n".encode('ascii'))
        
    trailer = f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode('ascii')
    body.append(trailer)
    
    with open(file_path, "wb") as f:
        f.write(b"".join(body))

# Seed data helper function
async def seed_sample_data():
    """
    Seeds the system with all sample data records required by the user prompt.
    """
    st.session_state.db.clear()
    st.session_state.expense_tool.expenses = []
    st.session_state.expense_tool._save_expenses()
    st.session_state.notes_tool.notes = []
    st.session_state.notes_tool._save_notes()
    st.session_state.memory_tool.clear()
    st.session_state.memory.clear()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc_dir = os.path.join(project_root, "documents")
    os.makedirs(doc_dir, exist_ok=True)
    
    elec_pdf = os.path.join(doc_dir, "electricity_bill.pdf")
    mri_pdf = os.path.join(doc_dir, "mri_report.pdf")
    
    create_minimal_pdf(elec_pdf, "Electricity Bill Details. Month: March 2026. Total Amount Due: INR 3500. Account: 928374.")
    create_minimal_pdf(mri_pdf, "Clinical Diagnosis Report. Scan: Brain MRI. Date: January 18, 2026. Findings: Patient shows normal brain scan.")
    
    await st.session_state.pdf_tool.ingest_pdf(elec_pdf)
    await st.session_state.pdf_tool.ingest_pdf(mri_pdf)
    
    await st.session_state.expense_tool.add_expense(
        description="Goa trip flight and hotel bookings", 
        amount=25000.0, 
        date_str="2026-02-10", 
        category="Travel"
    )
    await st.session_state.expense_tool.add_expense(
        description="Electricity bill payment", 
        amount=3500.0, 
        date_str="2026-03-01", 
        category="Utilities"
    )
    
    await st.session_state.notes_tool.add_note(
        content="EMI auto-debit date: 10th of every month.", 
        title="EMI Monthly Schedule"
    )
    await st.session_state.notes_tool.add_note(
        content="Goa trip dates: February 10 to February 15, 2026. Travelled with friends.", 
        title="Goa Trip Dates"
    )
    
    st.success("Sample data seeded successfully!")
    st.rerun()

def render_ui():
    st.set_page_config(
        page_title="Personal Memory & Life Admin Agent",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Injecting Custom Styles
    st.markdown("""
        <style>
        .main {
            background-color: #0F172A;
            color: #F8FAFC;
        }
        
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, #38BDF8, #818CF8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .metric-card {
            background: rgba(30, 41, 59, 0.7);
            border-radius: 12px;
            padding: 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .tool-badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 6px;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
            color: white;
        }
        .badge-pdf { background-color: #0284C7; }
        .badge-expense { background-color: #EA580C; }
        .badge-notes { background-color: #16A34A; }
        .badge-calc { background-color: #7C3AED; }
        .badge-mem { background-color: #EC4899; }
        
        .thought-card {
            background: #1E293B;
            border-left: 4px solid #38BDF8;
            padding: 0.8rem 1.2rem;
            border-radius: 4px;
            margin-bottom: 0.8rem;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session states
    if "db" not in st.session_state:
        with st.spinner("Loading Semantic Embedding Engine..."):
            st.session_state.db = FAISSDatabase()
            # Eagerly load the model to avoid timeouts during interactions
            _ = st.session_state.db.embedder.model
    if "pdf_tool" not in st.session_state:
        st.session_state.pdf_tool = PDFTool(st.session_state.db)
    if "expense_tool" not in st.session_state:
        st.session_state.expense_tool = ExpenseTool(st.session_state.db)
    if "notes_tool" not in st.session_state:
        st.session_state.notes_tool = NotesTool(st.session_state.db)
    if "calculator_tool" not in st.session_state:
        st.session_state.calculator_tool = CalculatorTool()
    if "memory_tool" not in st.session_state:
        st.session_state.memory_tool = MemoryTool(st.session_state.db)
    if "router" not in st.session_state:
        st.session_state.router = AgentRouter()
    if "planner" not in st.session_state:
        st.session_state.planner = AgentPlanner(
            pdf_tool=st.session_state.pdf_tool,
            expense_tool=st.session_state.expense_tool,
            notes_tool=st.session_state.notes_tool,
            calculator_tool=st.session_state.calculator_tool,
            memory_tool=st.session_state.memory_tool,
            router=st.session_state.router
        )
    if "memory" not in st.session_state:
        st.session_state.memory = HybridMemory(window_size=5)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()

    # Sidebar setup
    with st.sidebar:
        st.title("🧠 Settings & Stats")
        
        # API Configuration
        api_key = st.text_input("Gemini API Key (Optional)", type="password", help="Enables advanced multi-hop ReAct loop and conversation summarization.")
        
        st.write("---")
        st.subheader("📁 Ingest Records")
        
        # Load Sample Data Button
        if st.button("🔄 Seeding Sample Data", use_container_width=True, help="Load standard electricity bill, Goa trip, MRI report, and EMI schedule."):
            asyncio.run(seed_sample_data())
            
        # PDF Uploader
        uploaded_file = st.file_uploader("Upload Personal PDF Document", type=["pdf"])
        if uploaded_file is not None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            doc_dir = os.path.join(project_root, "documents")
            os.makedirs(doc_dir, exist_ok=True)
            save_path = os.path.join(doc_dir, uploaded_file.name)
            
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            with st.spinner(f"Indexing '{uploaded_file.name}'..."):
                res = asyncio.run(st.session_state.pdf_tool.ingest_pdf(save_path))
                if res["success"]:
                    st.success(res["message"])
                else:
                    st.error(res["message"])
                    
        st.write("---")
        st.subheader("📊 Memory Stats Panel")
        
        total_vectors = len(st.session_state.db.metadata)
        total_expenses = len(st.session_state.expense_tool.expenses)
        total_notes = len(st.session_state.notes_tool.notes)
        total_conversation = len(st.session_state.chat_history)
        total_facts = len(st.session_state.memory_tool.get_all_facts())
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Vector DB Records:</strong> {total_vectors} chunks<br>
            <strong>Expenses Logged:</strong> {total_expenses}<br>
            <strong>Personal Notes:</strong> {total_notes}<br>
            <strong>Conversation Count:</strong> {total_conversation} messages<br>
            <strong>Memory Facts:</strong> {total_facts} saved
        </div>
        """, unsafe_allow_html=True)
        
        st.write("---")
        st.subheader("👤 Remembered Profile Facts")
        facts = st.session_state.memory_tool.get_all_facts()
        if facts:
            facts_list_html = ""
            for k, v in facts.items():
                facts_list_html += f"<li><strong>{k.capitalize()}:</strong> {v}</li>"
            st.markdown(f"""
            <div class="metric-card" style="background: rgba(56, 189, 248, 0.05); border-left: 3px solid #38BDF8; padding: 1rem;">
                <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem; color: #E2E8F0; line-height: 1.4;">
                    {facts_list_html}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="font-size: 0.85rem; color: #94A3B8; font-style: italic;">
                No facts stored yet. Try saying: "My favourite food is Biryani" or "My pet name is Bruno".
            </div>
            """, unsafe_allow_html=True)
        
        st.write("---")
        
        # Clear Memory Button
        if st.button("🗑️ Clear Database & Memory", use_container_width=True, type="secondary"):
            st.session_state.db.clear()
            st.session_state.expense_tool.expenses = []
            st.session_state.expense_tool._save_expenses()
            st.session_state.notes_tool.notes = []
            st.session_state.notes_tool._save_notes()
            st.session_state.memory_tool.clear()
            st.session_state.memory.clear()
            st.session_state.chat_history = []
            save_chat_history([])
            st.success("All data and context cleared successfully!")
            st.rerun()

    # Main Chat Interface
    st.title("🧠 Personal Memory & Life Admin Agent")
    st.markdown("Retrieving knowledge, search logs, bills, and schedules through semantic memory planning.")

    # Render suggestion questions
    st.subheader("💡 Try Asking These Questions:")
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)
    
    preset_query = None
    if col1.button("What was my Goa trip budget?", use_container_width=True):
        preset_query = "What was my Goa trip budget?"
    if col2.button("When was my MRI report?", use_container_width=True):
        preset_query = "When was my MRI report?"
    if col3.button("When is my EMI due?", use_container_width=True):
        preset_query = "When is my EMI due?"
    if col4.button("How much was my electricity bill?", use_container_width=True):
        preset_query = "How much was my electricity bill?"
    if col5.button("How much money did I spend after my Goa trip? (Multi-hop)", use_container_width=True):
        preset_query = "How much money did I spend after my Goa trip?"

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "steps" in message and message["steps"]:
                with st.expander("Show Agent Reasoning", expanded=False):
                    for i, step in enumerate(message["steps"]):
                        st.markdown(f"**Step {i+1}:**")
                        # Badge mapping
                        badge_class = "badge-pdf"
                        action_name = step['action']
                        if "expense" in action_name:
                            badge_class = "badge-expense"
                        elif "note" in action_name:
                            badge_class = "badge-notes"
                        elif "calc" in action_name or "calculator" in action_name:
                            badge_class = "badge-calc"
                        elif "memory" in action_name:
                            badge_class = "badge-mem"
                            
                        st.markdown(f"""
                        <div class="thought-card">
                            <strong>Thought:</strong> {step['thought']}<br>
                            <strong>Action:</strong> <span class="tool-badge {badge_class}">{step['action']}</span><br>
                            <strong>Observation:</strong> {step['observation']}
                        </div>
                        """, unsafe_allow_html=True)

    # Chat input
    user_query = st.chat_input("Ask a question about notes, documents, or expenses...")
    if preset_query:
        user_query = preset_query

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        save_chat_history(st.session_state.chat_history)

        with st.chat_message("assistant"):
            with st.spinner("Agent planning and searching..."):
                # Collect contexts
                profile_context = st.session_state.memory_tool.get_formatted_context()
                memory_context = st.session_state.memory.get_formatted_context()
                
                parts = []
                if profile_context:
                    parts.append(profile_context)
                if memory_context:
                    parts.append(memory_context)
                    
                context = "\n\n".join(parts)
                full_query = f"{context}\n\nUser Query: {user_query}" if context else user_query
                
                # Execute graph workflow
                result = asyncio.run(
                    st.session_state.planner.execute(full_query, api_key=api_key)
                )
                answer = result["answer"]
                steps = result["steps"]
                
                st.write(answer)
                
                if steps:
                    with st.expander("Show Agent Reasoning", expanded=False):
                        for i, step in enumerate(steps):
                            st.markdown(f"**Step {i+1}:**")
                            # Badge mapping
                            badge_class = "badge-pdf"
                            action_name = step['action']
                            if "expense" in action_name:
                                badge_class = "badge-expense"
                            elif "note" in action_name:
                                badge_class = "badge-notes"
                            elif "calc" in action_name or "calculator" in action_name:
                                badge_class = "badge-calc"
                            elif "memory" in action_name:
                                badge_class = "badge-mem"
                                
                            st.markdown(f"""
                            <div class="thought-card">
                                <strong>Thought:</strong> {step['thought']}<br>
                                <strong>Action:</strong> <span class="tool-badge {badge_class}">{step['action']}</span><br>
                                <strong>Observation:</strong> {step['observation']}
                            </div>
                            """, unsafe_allow_html=True)
                            
            # Add dialogue turn to sliding hybrid memory
            st.session_state.memory.add_message("user", user_query, api_key=api_key)
            st.session_state.memory.add_message("assistant", answer, api_key=api_key)
            
            # Save assistant response to persistent UI log
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "steps": steps
            })
            save_chat_history(st.session_state.chat_history)
            
            if preset_query:
                st.rerun()

if __name__ == "__main__":
    render_ui()
