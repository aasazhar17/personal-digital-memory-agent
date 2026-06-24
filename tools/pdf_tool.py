import os
import asyncio
from typing import List, Dict, Any
from vectorstore.faiss_db import FAISSDatabase

class PDFTool:
    """
    Handles PDF document reading, chunking, and semantic vector database indexing/searching.
    """
    def __init__(self, db: FAISSDatabase):
        self.db = db

    async def ingest_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Asynchronously reads text from a PDF, splits it into semantic chunks, and adds it to the vector store.
        Uses run_in_executor to keep the main event loop responsive.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._ingest_pdf_sync, file_path)

    def _ingest_pdf_sync(self, file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            return {"success": False, "message": f"File '{file_path}' does not exist."}

        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            filename = os.path.basename(file_path)
            chunks = []
            metadatas = []
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue
                # Split text into page chunks
                page_chunks = self._chunk_text(text, chunk_size=600, overlap=100)
                for chunk in page_chunks:
                    chunks.append(chunk)
                    metadatas.append({
                        "source": filename,
                        "page": i + 1,
                        "type": "pdf"
                    })
            
            if chunks:
                self.db.add_texts(chunks, metadatas)
                return {
                    "success": True, 
                    "message": f"Successfully indexed {len(chunks)} text chunks from {filename}.",
                    "num_chunks": len(chunks)
                }
            else:
                return {"success": False, "message": f"No extractable text found in '{filename}'."}
        except Exception as e:
            return {"success": False, "message": f"Error indexing PDF '{file_path}': {str(e)}"}

    def _chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
        """
        Splits text into overlapping chunks.
        """
        chunks = []
        text = text.replace('\n', ' ').strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
            if start >= len(text) or (end == len(text)):
                break
        return chunks

    def match_filename(self, query: str) -> str:
        """
        Scans the documents directory and matches filenames to the query using:
        - lowercase conversion
        - removing spaces, &, _
        - word overlap checks
        """
        import re
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        doc_dir = os.path.join(project_root, "documents")
        if not os.path.exists(doc_dir):
            return None
            
        pdf_files = [f for f in os.listdir(doc_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            return None
            
        best_file = None
        best_score = 0.0
        
        q_clean = query.lower()
        q_norm = q_clean.replace(" ", "").replace("&", "").replace("_", "")
        q_words = set(re.findall(r'[a-z0-9]+', q_clean))
        
        for filename in pdf_files:
            score = 0.0
            fn_clean = filename.lower()
            fn_base = fn_clean.replace(".pdf", "")
            fn_norm = fn_base.replace(" ", "").replace("&", "").replace("_", "")
            fn_words = set(re.findall(r'[a-z0-9]+', fn_base))
            
            # 1. Exact base match or filename in query
            if fn_base in q_clean or filename in q_clean:
                score = 1.0
            # 2. Substring matching of normalized forms
            elif fn_norm in q_norm or q_norm in fn_norm:
                score = 0.9
            # 3. Word subset overlap
            elif fn_words.issubset(q_words) or q_words.issubset(fn_words):
                score = 0.8
            else:
                intersection = fn_words.intersection(q_words)
                # Ignore generic document search keywords
                non_generic = intersection - {"pdf", "document", "ticket", "invoice", "report", "bill", "receipt", "amount", "cost", "booking", "transaction", "pnr", "railway", "medical", "hospital", "travel", "show", "tell", "my"}
                if len(non_generic) >= 2:
                    score = 0.7
                elif len(non_generic) == 1:
                    score = 0.6
                    
            if score > best_score:
                best_score = score
                best_file = filename
                
        if best_score >= 0.6:
            return best_file
        return None

    async def search(self, query: str, k: int = 15) -> List[Dict[str, Any]]:
        """
        Asynchronously searches the vector store for matching PDF pages.
        Filters by fuzzy matched filename if found, otherwise falls back to general search.
        Removes duplicate text chunks and returns top 3 unique chunks.
        """
        import re
        matched_filename = self.match_filename(query)
        
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self.db.similarity_search, query, k)
        
        # Filter only PDF records
        pdf_results = [r for r in results if r.get("type") == "pdf"]
        
        # Filter by matched filename if any
        if matched_filename:
            pdf_results = [r for r in pdf_results if r.get("source", "").strip().lower() == matched_filename.lower()]
        else:
            # Fallback to regex check for specific filename
            pdf_match = re.search(r"\b([\w\-&]+\.pdf)\b", query, re.IGNORECASE)
            if pdf_match:
                filename = pdf_match.group(1).strip().lower()
                pdf_results = [r for r in pdf_results if r.get("source", "").strip().lower() == filename]
            
        # Remove duplicate text chunks
        unique_results = []
        seen_texts = set()
        for r in pdf_results:
            text_clean = r.get("text", "").strip()
            if text_clean not in seen_texts:
                seen_texts.add(text_clean)
                unique_results.append(r)
                
        return unique_results[:3]
