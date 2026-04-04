# Databricks notebook source

# MAGIC %md
# MAGIC # 03 — Build RAG Index
# MAGIC Ingests NHM health protocol PDFs, chunks them, and builds a FAISS vector index.
# MAGIC The index enables protocol-grounded Q&A for ASHA workers.

# COMMAND ----------

# MAGIC %pip install faiss-cpu sentence-transformers PyPDF2
# MAGIC %restart_python

# COMMAND ----------

import sys, os
repo_root = os.path.dirname(os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
))
sys.path.insert(0, f"/Workspace{repo_root}")

from src.rag_pipeline import RAGPipeline
from src.models.indic_llm import IndicLLM
from config.settings import LLM_PRIMARY_PATH, FAISS_INDEX_PATH, PROTOCOLS_DIR

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup Protocol Directory
# MAGIC
# MAGIC Upload NHM PDFs to DBFS first:
# MAGIC ```python
# MAGIC # Run this in a separate cell to upload PDFs from your local machine:
# MAGIC # dbutils.fs.cp("file:/path/to/pdf", "dbfs:/FileStore/asha_copilot/protocols/filename.pdf")
# MAGIC ```

# COMMAND ----------

# Ensure protocol directory exists
os.makedirs(PROTOCOLS_DIR, exist_ok=True)

# Check for PDFs
pdf_files = [f for f in os.listdir(PROTOCOLS_DIR) if f.endswith('.pdf')] if os.path.exists(PROTOCOLS_DIR) else []
print(f"Found {len(pdf_files)} PDF files in {PROTOCOLS_DIR}")
for f in pdf_files:
    size_mb = os.path.getsize(os.path.join(PROTOCOLS_DIR, f)) / (1024**2)
    print(f"  - {f} ({size_mb:.1f} MB)")

# Also check repo-local data directory
repo_pdf_dir = f"/Workspace{repo_root}/data/nhm_protocols"
repo_pdfs = [f for f in os.listdir(repo_pdf_dir) if f.endswith('.pdf')] if os.path.exists(repo_pdf_dir) else []
if repo_pdfs:
    print(f"\nAlso found {len(repo_pdfs)} PDFs in repo: {repo_pdf_dir}")
    # Copy to DBFS
    for f in repo_pdfs:
        src = os.path.join(repo_pdf_dir, f)
        dst = os.path.join(PROTOCOLS_DIR, f)
        if not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)
            print(f"  Copied {f} to DBFS")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize RAG Pipeline

# COMMAND ----------

# Load LLM for answer generation
llm = IndicLLM(model_path=LLM_PRIMARY_PATH, n_ctx=2048, n_threads=4)
rag = RAGPipeline(llm=llm)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ingest PDFs & Build Index

# COMMAND ----------

# Try DBFS protocols dir first, fall back to repo dir
pdf_dir = PROTOCOLS_DIR if pdf_files else repo_pdf_dir

n_chunks = rag.ingest_pdfs(pdf_dir)
print(f"Total chunks created: {n_chunks}")

if n_chunks > 0:
    rag.build_index()
    rag.save_index(FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")
else:
    print("\nNo PDFs found. To use RAG, upload NHM protocol PDFs:")
    print(f"  1. Upload to: {PROTOCOLS_DIR}")
    print("  2. Re-run this notebook")
    print("\nContinuing with demo mode (RAG will return 'no data' responses)...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test RAG Queries

# COMMAND ----------

test_queries = [
    "What are the danger signs during pregnancy?",
    "What is the catch-up schedule for a missed Pentavalent dose?",
    "When should a newborn be referred to a health facility?",
    "What are the HBNC visit schedules after birth?",
    "What is the treatment for severe anemia in pregnancy?",
]

for query in test_queries:
    print(f"\nQ: {query}")
    print("-" * 60)
    result = rag.answer(query)
    print(f"A: {result['answer'][:300]}...")
    if result['sources']:
        print(f"Sources: {', '.join(result['sources'])}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Index Summary

# COMMAND ----------

if os.path.exists(FAISS_INDEX_PATH):
    print(f"FAISS index location: {FAISS_INDEX_PATH}")
    for f in os.listdir(FAISS_INDEX_PATH):
        fpath = os.path.join(FAISS_INDEX_PATH, f)
        size_mb = os.path.getsize(fpath) / (1024**2)
        print(f"  {f}: {size_mb:.2f} MB")
else:
    print("No index persisted (no PDFs were available)")
