import asyncio
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from vectorstore.faiss_db import FAISSDatabase
from tools.pdf_tool import PDFTool
from tools.expense_tool import ExpenseTool
from tools.notes_tool import NotesTool
from tools.memory_tool import MemoryTool
from memory.hybrid_memory import HybridMemory

def create_minimal_pdf(file_path: str, text: str):
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

    add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")
    add_obj(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    add_obj(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")
    stream_content = f"BT /F1 12 Tf 72 712 Td ({text}) Tj ET".encode('latin-1')
    stream_len = len(stream_content)
    add_obj(f"<< /Length {stream_len} >>\nstream\n".encode('ascii') + stream_content + b"\nendstream")
    add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    
    xref_offset = sum(len(x) for x in body)
    xref_header = f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f\n".encode('ascii')
    body.append(xref_header)
    
    for off in offsets:
        body.append(f"{off:010d} 00000 n\n".encode('ascii'))
        
    trailer = f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode('ascii')
    body.append(trailer)
    
    with open(file_path, "wb") as f:
        f.write(b"".join(body))

async def main():
    print("Initializing FAISS Vector Store...")
    db = FAISSDatabase()
    db.clear()
    
    pdf_tool = PDFTool(db)
    expense_tool = ExpenseTool(db)
    notes_tool = NotesTool(db)
    memory_tool = MemoryTool(db)
    memory = HybridMemory(window_size=5)
    
    # Clear memory profile and cache
    expense_tool.expenses = []
    expense_tool._save_expenses()
    notes_tool.notes = []
    notes_tool._save_notes()
    memory_tool.clear()
    memory.clear()
    
    print("Creating and Ingesting Electricity Bill PDF...")
    doc_dir = os.path.join(project_root, "documents")
    os.makedirs(doc_dir, exist_ok=True)
    
    elec_pdf = os.path.join(doc_dir, "electricity_bill.pdf")
    create_minimal_pdf(elec_pdf, "Electricity Bill Details. Month: March 2026. Total Amount Due: INR 3500. Account: 928374.")
    res_elec = await pdf_tool.ingest_pdf(elec_pdf)
    print("Electricity PDF Ingestion:", res_elec)
    
    print("Creating and Ingesting Brain MRI Report PDF...")
    mri_pdf = os.path.join(doc_dir, "mri_report.pdf")
    create_minimal_pdf(mri_pdf, "Clinical Diagnosis Report. Scan: Brain MRI. Date: January 18, 2026. Findings: Patient shows normal brain scan.")
    res_mri = await pdf_tool.ingest_pdf(mri_pdf)
    print("MRI PDF Ingestion:", res_mri)
    
    print("Logging Goa Trip Expenses...")
    await expense_tool.add_expense(
        description="Goa trip flight and hotel bookings", 
        amount=25000.0, 
        date_str="2026-02-10", 
        category="Travel"
    )
    await expense_tool.add_expense(
        description="Electricity bill payment", 
        amount=3500.0, 
        date_str="2026-03-01", 
        category="Utilities"
    )
    
    print("Creating EMI and Goa Trip Notes...")
    await notes_tool.add_note(
        content="EMI auto-debit date: 10th of every month.", 
        title="EMI Monthly Schedule"
    )
    await notes_tool.add_note(
        content="Goa trip dates: February 10 to February 15, 2026. Travelled with friends.", 
        title="Goa Trip Dates"
    )
    
    print("Saving User Profile Memory Facts...")
    memory_tool.store_fact("name", "Azhar")
    memory_tool.store_fact("city", "Delhi")
    memory_tool.store_fact("college", "ABC University")
    memory_tool.store_fact("pet name", "Bruno")
    memory_tool.store_fact("favourite food", "Biryani")
    
    print("\nDatabase and memory initialized successfully!")

if __name__ == "__main__":
    asyncio.run(main())
