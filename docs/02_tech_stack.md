# Tech Stack

## Overview

Asha AI is built on the **Databricks Lakehouse Platform** with a focus on leveraging Databricks-native services for data storage, compute, security, and deployment — complemented by open-source Indic AI models and Sarvam AI APIs for multilingual voice interaction.

---

## Databricks Technologies

| Technology | Usage in Asha AI | Details |
|-----------|------------------|---------|
| **Databricks Workspace** | Primary development and deployment environment | Notebooks, repos, compute clusters, apps |
| **Unity Catalog** | Data governance and schema management | `workspace.asha_copilot` catalog with 7 managed Delta tables |
| **Delta Lake** | Transactional data storage for all patient, visit, triage, and immunization records | ACID-compliant, schema-enforced, time-travel enabled |
| **Databricks SQL** | Analytical queries powering the supervisor dashboard | Spark SQL over Delta tables for real-time aggregations |
| **Databricks Repos** | Version-controlled notebooks synced from GitHub | Git integration for collaborative development |
| **Databricks Apps** | Production deployment of the Gradio-based copilot UI | Containerized app runtime via `app.yaml` manifest |
| **Databricks Secret Scopes** | Secure credential management for API keys | Scope `asha-ai` stores Sarvam, HuggingFace, and Groq API keys |
| **Databricks Connect** | Remote Spark session access from the app runtime | `databricks-connect>=15.4.0` for Delta table operations |
| **DBFS (Databricks File System)** | Persistent storage for NHM protocol PDFs and FAISS index | `/dbfs/FileStore/asha_copilot/protocols/` |
| **Serverless Compute** | On-demand Spark sessions with zero cluster management | Used for notebook execution and app runtime |

### Delta Lake Schema: `workspace.asha_copilot`

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `patients` | Patient demographics | patient_id, name, age, village, district, bpl_status, aadhaar |
| `visits` | Clinical visit records | visit_id, patient_id, bp_systolic, bp_diastolic, hemoglobin, symptoms, visit_date |
| `immunizations` | Vaccine administration records | patient_id, vaccine_name, dose_number, date_given, date_due |
| `triage_alerts` | Risk classification results | patient_id, risk_level, sign_name, action, assigned_doctor, urgency_hours |
| `doctors` | Doctor registry | doctor_id, name, specialization, phone, phc_id |
| `phc_facilities` | Primary Health Centre directory | phc_id, name, type, district, latitude, longitude |
| `scheme_applications` | Welfare scheme enrollment status | patient_id, scheme_name, eligible, missing_requirements, amount |

---

## Open-Source Models & AI APIs

### Primary AI Stack (Sarvam AI)

| Component | Model / API | Provider | Purpose |
|-----------|------------|----------|---------|
| **LLM** | Sarvam-m (30B multilingual) | Sarvam AI API | Intent classification, entity extraction, clinical reasoning, RAG grounding |
| **Translation** | Mayura | Sarvam AI API | Hindi ↔ English bidirectional translation |
| **Speech-to-Text** | Saarika (Saaras v3) | Sarvam AI API | Hindi/English automatic speech recognition |
| **Text-to-Speech** | Bulbul v2 | Sarvam AI API | Natural Hindi voice synthesis |

### Fallback Stack (Open-Source / HuggingFace)

| Component | Model | Provider | Purpose |
|-----------|-------|----------|---------|
| **LLM (Fallback)** | Llama 3.3 70B Versatile | Groq API | High-throughput LLM when Sarvam unavailable |
| **LLM (Fallback 2)** | HuggingFace Inference Router | HuggingFace | Final fallback for LLM inference |
| **Translation (Fallback)** | IndicTrans2 | HuggingFace | Open-source Hindi ↔ English translation |
| **ASR (Fallback)** | IndicWhisper | HuggingFace | Open-source Hindi speech recognition |
| **TTS (Fallback)** | gTTS (Google Text-to-Speech) | Google | Basic TTS when Sarvam unavailable |

### Embedding & Vector Search

| Component | Technology | Details |
|-----------|-----------|---------|
| **Embedding Model** | `all-MiniLM-L6-v2` (sentence-transformers) | 384-dimensional embeddings, ~80 MB on disk, ~200 MB RAM |
| **Vector Database** | FAISS (CPU) | In-memory index, persisted to DBFS, ~5 MB for NHM protocols |
| **PDF Parser** | PyPDF2 3.0.1 | Extracts text from NHM/SUMAN clinical protocol PDFs |
| **Chunking Strategy** | 500-token chunks, 50-token overlap | Optimized for clinical guideline passages |

---

## Frontend & Visualization

| Technology | Usage | Details |
|-----------|-------|---------|
| **Gradio 4.44** | Primary UI framework | Tabbed interface with voice recording, forms, chatbot, and data tables |
| **Web Audio API** | Browser-side voice recording | Custom JavaScript for 16 kHz mono WAV capture, base64 encoding |
| **Plotly** | Interactive dashboard charts | Pie charts, stacked bar charts, line charts for supervisor analytics |
| **Pandas** | Data transformation | DataFrame operations for dashboard and display components |

---

## Infrastructure & DevOps

| Component | Technology | Details |
|-----------|-----------|---------|
| **Runtime** | Python 3.10+ | All backend logic in Python |
| **Package Management** | pip + `requirements.txt` | 12 core dependencies |
| **Testing** | pytest | Unit tests for triage engine (`tests/test_triage.py`) |
| **Configuration** | YAML + Python | `danger_signs.yaml`, `phc_doctors.yaml`, `config/settings.py` |
| **Secret Management** | Databricks Secret Scopes | Scope: `asha-ai` with keys for Sarvam, HF, Groq |
| **HTTP Client** | requests 2.31 | API communication with Sarvam AI, Groq, HuggingFace |

---

## Resource Budget

| Component | Disk | RAM | GPU | Notes |
|-----------|------|-----|-----|-------|
| Sarvam AI APIs (LLM + Translation + ASR + TTS) | 0 GB | 0 GB | None | All inference via API |
| all-MiniLM-L6-v2 | ~80 MB | ~200 MB | None | Auto-downloaded on first use |
| FAISS index | ~5 MB | ~50 MB | None | Scales with protocol corpus size |
| **Total** | **~85 MB** | **~250 MB** | **None** | Runs on any Databricks cluster or local machine |

> *"If your solution only works on an A100, it does not work in India."*

---

## Databricks Technologies Used (Summary)

- Delta Lake (ACID-compliant data storage)
- Unity Catalog (data governance)
- Databricks SQL (analytical queries)
- Databricks Apps (production deployment)
- Databricks Repos (Git-integrated notebooks)
- Databricks Secret Scopes (credential management)
- Databricks Connect (remote Spark access)
- DBFS (file storage)
- Serverless Compute (on-demand Spark)
