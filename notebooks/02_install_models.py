# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Setup Models & API Connectivity
# MAGIC Installs Python dependencies, sets API keys, and verifies connectivity.
# MAGIC **No large model downloads required** — LLM/translation/TTS use Sarvam AI API.
# MAGIC
# MAGIC | Component | How | Size |
# MAGIC |-----------|-----|------|
# MAGIC | LLM (sarvam-m) | Sarvam AI API | 0 GB local |
# MAGIC | Translation (Mayura) | Sarvam AI API | 0 GB local |
# MAGIC | TTS (Bulbul v2) | Sarvam AI API | 0 GB local |
# MAGIC | ASR (Saarika) | Sarvam AI API | 0 GB local |
# MAGIC | RAG Embeddings | sentence-transformers (auto-download ~80 MB) | ~80 MB |
# MAGIC | FAISS index | Built in notebook 03 | ~5 MB |

# COMMAND ----------

# MAGIC %pip install sentence-transformers faiss-cpu huggingface_hub requests gtts
# MAGIC %restart_python

# COMMAND ----------

import sys, os

repo_root = os.path.dirname(os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
))
sys.path.insert(0, f"/Workspace{repo_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Set API Keys
# MAGIC Set these in your cluster environment variables for persistence across restarts,
# MAGIC or run this cell each session.

# COMMAND ----------

import os

# Set your API keys here (or configure in cluster Environment Variables)
os.environ["SARVAM_API_KEY"] = dbutils.secrets.get(scope="asha-copilot", key="sarvam-api-key") if False else os.environ.get("SARVAM_API_KEY", "")
os.environ["HF_TOKEN"] = dbutils.secrets.get(scope="asha-copilot", key="hf-token") if False else os.environ.get("HF_TOKEN", "")

print(f"SARVAM_API_KEY set: {'YES' if os.environ.get('SARVAM_API_KEY') else 'NO — set it above!'}")
print(f"HF_TOKEN set:       {'YES' if os.environ.get('HF_TOKEN') else 'NO (optional fallback)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Test Sarvam LLM API

# COMMAND ----------

from src.models.indic_llm import IndicLLM

llm = IndicLLM()
print(f"LLM API available: {llm.is_loaded}")

test_response = llm.generate(
    "You are a healthcare assistant. What are the 3 most common danger signs during pregnancy? "
    "Answer in 2 sentences.",
    max_tokens=150,
)
print(f"\nTest response:\n{test_response}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test Intent Classification

# COMMAND ----------

test_inputs = [
    "Register new patient Sunita Devi age 24 from Rampur village",
    "BP is 150/95, hemoglobin 7.5, she has severe headache",
    "Is this patient eligible for JSY scheme?",
    "When is the next Pentavalent vaccine due?",
    "What are the HBNC visit schedules?",
]

print("Intent Classification Test:")
print("-" * 60)
for text in test_inputs:
    intent = llm.classify_intent(text)
    print(f"  [{intent:20s}] ← {text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test Sarvam Translation API

# COMMAND ----------

from src.models.indic_translate import IndicTranslator

translator = IndicTranslator()
print(f"Translator available: {translator.is_loaded}")

test_hindi = "मरीज को तेज बुखार है और सांस लेने में तकलीफ है।"
translated = translator.to_english(test_hindi)
print(f"\nHindi → English:")
print(f"  Input:  {test_hindi}")
print(f"  Output: {translated}")

back = translator.from_english("The patient has high fever and difficulty breathing.", "hi")
print(f"\nEnglish → Hindi:")
print(f"  Input:  The patient has high fever and difficulty breathing.")
print(f"  Output: {back}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify RAG Embedding Model

# COMMAND ----------

from sentence_transformers import SentenceTransformer

print("Loading sentence-transformers/all-MiniLM-L6-v2...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

test_embedding = embedder.encode(["What are the danger signs in pregnancy?"])
print(f"Embedding model loaded. Vector dim: {test_embedding.shape[1]}")
print(f"Test vector (first 5): {test_embedding[0][:5].tolist()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Summary

# COMMAND ----------

print("=" * 60)
print("SETUP SUMMARY")
print("=" * 60)
print(f"  Sarvam AI API (LLM sarvam-m):   {'✓ Connected' if llm.is_loaded else '✗ No API key'}")
print(f"  Sarvam Translation (Mayura):     {'✓ Connected' if translator.is_loaded else '✗ No API key'}")
print(f"  Sentence Embeddings (MiniLM):    ✓ Loaded ({test_embedding.shape[1]}-dim)")
print()
print("Next step: Run notebook 03_build_rag_index.py")
print("  (upload NHM PDFs to data/nhm_protocols/ first for best RAG results)")
