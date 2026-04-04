"""
Central configuration for ASHA Copilot.
All configurable values in one place — no hardcoded strings in notebooks or src modules.
"""

# ---------------------------------------------------------------------------
# Database (hive_metastore — Free Edition default catalog)
# ---------------------------------------------------------------------------
CATALOG = "hive_metastore"
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
# Model paths on DBFS (downloaded by notebook 02_install_models)
# ---------------------------------------------------------------------------
MODELS_DIR = "/dbfs/FileStore/asha_copilot/models"
FAISS_INDEX_PATH = "/dbfs/FileStore/asha_copilot/faiss_index"
PROTOCOLS_DIR = "/dbfs/FileStore/asha_copilot/protocols"

# Primary LLM — Airavata 7B GGUF Q4_K_M (Hindi instruction-tuned)
LLM_PRIMARY_MODEL = "Airavata-7B-Q4_K_M.gguf"
LLM_PRIMARY_PATH = f"{MODELS_DIR}/{LLM_PRIMARY_MODEL}"
LLM_PRIMARY_CTX = 2048  # context window tokens
LLM_PRIMARY_THREADS = 4  # CPU threads for llama.cpp

# Lightweight LLM — Param-1 2.9B Q4 (intent classification, entity extraction)
LLM_LIGHT_MODEL = "Param-1-2.9B-Q4.gguf"
LLM_LIGHT_PATH = f"{MODELS_DIR}/{LLM_LIGHT_MODEL}"
LLM_LIGHT_CTX = 1024

# Translation — IndicTrans2 ONNX 200M
TRANSLATE_MODEL_DIR = f"{MODELS_DIR}/indictrans2"

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
# API fallback endpoints (used if local GGUF models fail to load)
# ---------------------------------------------------------------------------
SARVAM_API_URL = "https://api.sarvam.ai"
AI4BHARAT_API_URL = "https://api.ai4bharat.org"
HF_INFERENCE_API = "https://api-inference.huggingface.co/models"

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
