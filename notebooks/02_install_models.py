# Databricks notebook source

# MAGIC %md
# MAGIC # 02 — Install & Cache ML Models
# MAGIC Downloads quantized Indic models to DBFS for CPU inference.
# MAGIC **Idempotent** — skips already downloaded models.
# MAGIC
# MAGIC | Model | Size | Purpose |
# MAGIC |-------|------|---------|
# MAGIC | Airavata 7B Q4_K_M | ~4 GB | Hindi clinical reasoning, triage |
# MAGIC | sentence-transformers/all-MiniLM-L6-v2 | ~80 MB | RAG embeddings |
# MAGIC
# MAGIC **Note**: Model downloads happen once. Subsequent runs use cached files from DBFS.

# COMMAND ----------

# MAGIC %pip install llama-cpp-python ctransformers sentence-transformers faiss-cpu huggingface_hub
# MAGIC %restart_python

# COMMAND ----------

import os

MODELS_DIR = "/dbfs/FileStore/asha_copilot/models"
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"Models directory: {MODELS_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Download Airavata 7B GGUF (Hindi Instruction-Tuned LLM)

# COMMAND ----------

from huggingface_hub import hf_hub_download

airavata_path = os.path.join(MODELS_DIR, "Airavata-7B-Q4_K_M.gguf")

if os.path.exists(airavata_path):
    size_gb = os.path.getsize(airavata_path) / (1024**3)
    print(f"Airavata already cached: {airavata_path} ({size_gb:.1f} GB)")
else:
    print("Downloading Airavata 7B GGUF (Q4_K_M)... This may take 10-15 minutes.")
    # Try downloading from a GGUF-converted repo on HuggingFace
    # If this specific repo doesn't exist, users can substitute any Hindi-capable GGUF model
    try:
        downloaded = hf_hub_download(
            repo_id="ai4bharat/Airavata",
            filename="airavata-7b-q4_k_m.gguf",
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded to: {downloaded}")
    except Exception as e:
        print(f"Direct GGUF download failed: {e}")
        print("Attempting alternative: downloading from TheBloke's quantized models...")
        try:
            downloaded = hf_hub_download(
                repo_id="TheBloke/OpenHathi-7B-Hi-v0.1-Base-GGUF",
                filename="openhathi-7b-hi-v0.1-base.Q4_K_M.gguf",
                local_dir=MODELS_DIR,
                local_dir_use_symlinks=False,
            )
            # Rename for consistency
            os.rename(downloaded, airavata_path)
            print(f"Downloaded and renamed to: {airavata_path}")
        except Exception as e2:
            print(f"Alternative download also failed: {e2}")
            print("You can manually upload a GGUF model to:", airavata_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Verify Embedding Model (auto-downloads on first use)

# COMMAND ----------

from sentence_transformers import SentenceTransformer

print("Loading sentence-transformers/all-MiniLM-L6-v2...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Test embedding
test_embedding = embedder.encode(["What are the danger signs in pregnancy?"])
print(f"Embedding model loaded. Vector dim: {test_embedding.shape[1]}")
print(f"Test vector (first 5 values): {test_embedding[0][:5]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test LLM Loading

# COMMAND ----------

import sys, os
repo_root = os.path.dirname(os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
))
sys.path.insert(0, f"/Workspace{repo_root}")

from src.models.indic_llm import IndicLLM

llm = IndicLLM(model_path=airavata_path, n_ctx=2048, n_threads=4)

if llm.is_loaded:
    print("LLM loaded successfully!")
    # Quick test
    test_response = llm.generate("What are the danger signs during pregnancy?", max_tokens=100)
    print(f"Test response: {test_response[:200]}...")
else:
    print("LLM not loaded locally — will use API fallback.")
    print("Set HF_TOKEN environment variable for Hugging Face Inference API access.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test Intent Classification

# COMMAND ----------

test_inputs = [
    "Register new patient Sunita Devi age 24 from Rampur village",
    "BP is 150/95, hemoglobin 7.5, she has severe headache",
    "Is this patient eligible for JSY scheme?",
    "When is the next Pentavalent vaccine due?",
    "What are the HBNC visit schedules?",
]

for text in test_inputs:
    intent = llm.classify_intent(text)
    print(f"  [{intent:20s}] ← {text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Cache Summary

# COMMAND ----------

print("=" * 60)
print("MODEL CACHE STATUS")
print("=" * 60)
for fname in os.listdir(MODELS_DIR):
    fpath = os.path.join(MODELS_DIR, fname)
    if os.path.isfile(fpath):
        size_mb = os.path.getsize(fpath) / (1024**2)
        print(f"  {fname:50s} → {size_mb:,.1f} MB")
    elif os.path.isdir(fpath):
        print(f"  {fname:50s} → [directory]")
print(f"\nLLM loaded: {llm.is_loaded}")
print(f"Embedder dim: {test_embedding.shape[1]}")
