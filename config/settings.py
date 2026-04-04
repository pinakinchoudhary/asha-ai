"""
Central configuration for ASHA Copilot.
All configurable values in one place — no hardcoded strings in notebooks or src modules.
"""

import os

# ---------------------------------------------------------------------------
# Database (Unity Catalog — workspace catalog on Databricks)
# ---------------------------------------------------------------------------
CATALOG = "workspace"
DATABASE = "asha_copilot"
FULL_DB = f"{CATALOG}.{DATABASE}"

# Table names
TABLE_PATIENTS = f"{FULL_DB}.patients"
TABLE_VISITS = f"{FULL_DB}.visits"
TABLE_IMMUNIZATIONS = f"{FULL_DB}.immunizations"
TABLE_SCHEME_APPLICATIONS = f"{FULL_DB}.scheme_applications"
TABLE_TRIAGE_ALERTS = f"{FULL_DB}.triage_alerts"
TABLE_DOCTORS = f"{FULL_DB}.doctors"
TABLE_PHC_FACILITIES = f"{FULL_DB}.phc_facilities"

# ---------------------------------------------------------------------------
# File paths — derived relative to repo root (works on Databricks Workspace)
# ---------------------------------------------------------------------------
_SETTINGS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SETTINGS_DIR)

# Where sentence-transformers cache is stored (auto-managed by HuggingFace)
MODELS_DIR = os.path.join(_REPO_ROOT, "models")

# FAISS index persisted here after notebook 03
FAISS_INDEX_PATH = os.path.join(_REPO_ROOT, "data", "faiss_index")

# NHM protocol PDFs — upload here manually or via dbutils.fs.cp
PROTOCOLS_DIR = os.path.join(_REPO_ROOT, "data", "nhm_protocols")

# Legacy GGUF paths (kept for reference; not used — Sarvam API is primary)
LLM_PRIMARY_PATH = None
TRANSLATE_MODEL_DIR = None

# ---------------------------------------------------------------------------
# Sarvam AI API (primary LLM, translation, TTS)
# Set SARVAM_API_KEY in environment or Databricks cluster config
# ---------------------------------------------------------------------------
SARVAM_API_BASE = "https://api.sarvam.ai"
SARVAM_LLM_MODEL = "sarvam-m"

# HuggingFace Inference API (fallback for ASR + LLM if Sarvam unavailable)
HF_INFERENCE_API = "https://api-inference.huggingface.co/models"

# Embeddings — sentence-transformers (auto-downloaded by HuggingFace)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# Clinical thresholds
# ---------------------------------------------------------------------------
TRIAGE_CONFIDENCE_THRESHOLD = 0.85
BP_SYSTOLIC_HIGH = 140
BP_DIASTOLIC_HIGH = 90
BP_SYSTOLIC_CRITICAL = 160
BP_DIASTOLIC_CRITICAL = 110
HB_SEVERE_ANEMIA = 7.0
HB_MODERATE_ANEMIA = 9.0
TEMP_HIGH_FEVER = 38.5  # Celsius

# ---------------------------------------------------------------------------
# Supported languages (Hindi + English for demo)
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {"hi": "Hindi", "en": "English"}
DEFAULT_LANGUAGE = "hi"

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
SYNTH_NUM_PATIENTS = 500
SYNTH_NUM_PHC = 10
SYNTH_NUM_DOCTORS = 30
SYNTH_DANGER_SIGN_RATE = 0.07  # 5-8% of visits
SYNTH_BPL_RATE = 0.30  # ~30% BPL households
SYNTH_ANEMIA_RATE = 0.55  # ~55% mild-moderate anemia per NFHS-5

# ---------------------------------------------------------------------------
# RAG chunking
# ---------------------------------------------------------------------------
RAG_CHUNK_SIZE = 500  # tokens
RAG_CHUNK_OVERLAP = 50
RAG_TOP_K = 3
