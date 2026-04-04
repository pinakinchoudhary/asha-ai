"""
RAG Pipeline — Retrieval-Augmented Generation over NHM health protocol PDFs.

Uses FAISS (CPU) + sentence-transformers for vector search.
LLM answers grounded in retrieved protocol chunks.
"""

import logging
import os
import pickle

logger = logging.getLogger(__name__)


class RAGPipeline:
    """PDF ingestion → chunking → FAISS index → retrieval → LLM-grounded answers."""

    def __init__(self, llm=None, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.llm = llm
        self._embedding_model_name = embedding_model
        self._embedder = None
        self._index = None
        self._chunks = []  # parallel list of text chunks

    def _load_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embedding_model_name)
            logger.info(f"Loaded embedding model: {self._embedding_model_name}")

    # ------------------------------------------------------------------
    # PDF ingestion + chunking
    # ------------------------------------------------------------------

    def ingest_pdfs(self, pdf_dir: str) -> int:
        """
        Extract text from all PDFs in directory and chunk them.
        Returns number of chunks created.
        """
        from PyPDF2 import PdfReader

        all_text = []
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return 0

        for pdf_file in pdf_files:
            path = os.path.join(pdf_dir, pdf_file)
            try:
                reader = PdfReader(path)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        all_text.append({
                            "text": text.strip(),
                            "source": pdf_file,
                            "page": page_num + 1,
                        })
            except Exception as e:
                logger.warning(f"Failed to read {pdf_file}: {e}")

        # Chunk the extracted text
        self._chunks = []
        for doc in all_text:
            chunks = self._chunk_text(doc["text"], chunk_size=500, overlap=50)
            for i, chunk in enumerate(chunks):
                self._chunks.append({
                    "text": chunk,
                    "source": doc["source"],
                    "page": doc["page"],
                    "chunk_index": i,
                })

        logger.info(f"Ingested {len(pdf_files)} PDFs → {len(self._chunks)} chunks")
        return len(self._chunks)

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - overlap
        return chunks

    # ------------------------------------------------------------------
    # FAISS index build + persistence
    # ------------------------------------------------------------------

    def build_index(self):
        """Build FAISS index from ingested chunks."""
        if not self._chunks:
            logger.warning("No chunks to index. Run ingest_pdfs first.")
            return

        import numpy as np
        import faiss

        self._load_embedder()

        texts = [c["text"] for c in self._chunks]
        embeddings = self._embedder.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embeddings)

        logger.info(f"FAISS index built: {self._index.ntotal} vectors, dim={dim}")

    def save_index(self, path: str):
        """Persist FAISS index + chunks to disk."""
        import faiss

        os.makedirs(path, exist_ok=True)
        if self._index:
            faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info(f"Index saved to {path}")

    def load_index(self, path: str) -> bool:
        """Load persisted FAISS index + chunks."""
        import faiss

        index_path = os.path.join(path, "index.faiss")
        chunks_path = os.path.join(path, "chunks.pkl")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            logger.warning(f"No persisted index found at {path}")
            return False

        self._index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self._chunks = pickle.load(f)

        self._load_embedder()
        logger.info(f"Loaded index: {self._index.ntotal} vectors, {len(self._chunks)} chunks")
        return True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Retrieve top-k relevant chunks for a query."""
        if not self._index or not self._chunks:
            logger.warning("No index available. Build or load index first.")
            return []

        import numpy as np

        self._load_embedder()
        query_vec = self._embedder.encode([query])
        query_vec = np.array(query_vec, dtype="float32")

        distances, indices = self._index.search(query_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._chunks):
                chunk = self._chunks[idx].copy()
                chunk["distance"] = float(dist)
                results.append(chunk)
        return results

    # ------------------------------------------------------------------
    # RAG answer generation
    # ------------------------------------------------------------------

    def answer(self, question: str, top_k: int = 3) -> dict:
        """
        Retrieve relevant chunks and generate a grounded answer.
        Returns dict with answer, sources, and retrieved chunks.
        """
        chunks = self.retrieve(question, top_k=top_k)

        if not chunks:
            return {
                "answer": "I don't have enough information to answer this question. "
                          "Please consult your supervisor or refer to the NHM guidelines.",
                "sources": [],
                "chunks": [],
            }

        context = "\n\n".join([
            f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
            for c in chunks
        ])

        if self.llm:
            prompt = (
                "You are an ASHA health assistant trained on NHM (National Health Mission) "
                "protocols. Using ONLY the following protocol excerpts, answer the health "
                "worker's question accurately and concisely. If the answer is not in the "
                "excerpts, say 'I don't have this information in the available protocols. "
                "Please consult your supervisor.'\n\n"
                f"Protocol excerpts:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )
            answer_text = self.llm.generate(prompt, max_tokens=300, temperature=0.2)
        else:
            # No LLM — return raw chunks
            answer_text = (
                "Based on the available protocols:\n\n"
                + "\n\n".join([c["text"][:200] + "..." for c in chunks])
            )

        sources = list(set(f"{c['source']} (Page {c['page']})" for c in chunks))

        return {
            "answer": answer_text,
            "sources": sources,
            "chunks": chunks,
        }
