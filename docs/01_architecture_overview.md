# Architecture Overview

## What It Does

Asha AI is a voice-first, multilingual field assistant that empowers India's 1 million+ Accredited Social Health Activists (ASHAs) with AI-driven clinical triage, NHM protocol guidance, welfare scheme discovery, immunization tracking, and voice-driven patient management — all running on CPU with open-source Indic AI models atop the Databricks Lakehouse platform.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        ASHA Worker (Mobile)                         │
│                     Hindi / English Voice Input                      │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                               │
│                                                                      │
│   Gradio UI (Databricks Apps / In-Notebook)                         │
│   ├── Voice Copilot Tab          ├── Triage Tab                     │
│   ├── Patient CRUD Tab           ├── Scheme Checker Tab             │
│   └── Protocol Q&A Tab           └── Supervisor Dashboard           │
│                                                                      │
│   Web Audio API (16 kHz mono WAV, browser-side JS recording)        │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     SPEECH & LANGUAGE LAYER                           │
│                                                                      │
│   ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│   │ Sarvam Saarika   │  │ Sarvam Mayura    │  │ Sarvam Bulbul   │   │
│   │ ASR (STT)        │  │ Translation      │  │ TTS             │   │
│   │ Hindi Speech →   │  │ Hindi ↔ English  │  │ English/Hindi → │   │
│   │ Hindi Text       │  │ Bidirectional    │  │ Hindi Audio     │   │
│   └────────┬─────────┘  └────────┬─────────┘  └────────┬────────┘   │
│            │  Fallback: HF       │  Fallback: HF       │ Fallback:  │
│            │  IndicWhisper       │  IndicTrans2         │ gTTS       │
└────────────┼─────────────────────┼─────────────────────┼────────────┘
             │                     │                     │
             ▼                     ▼                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     INTELLIGENCE LAYER                                │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  LLM (Groq API — Llama 3.3 70B / Sarvam-m)                 │   │
│   │  Intent Classification → Route to Appropriate Engine        │   │
│   └──┬──────────┬──────────┬──────────┬──────────┬──────────────┘   │
│      │          │          │          │          │                    │
│   ┌──▼───┐  ┌──▼───┐  ┌──▼───┐  ┌──▼───┐  ┌──▼────────┐          │
│   │Triage│  │Voice │  │Scheme│  │ RAG  │  │Immuniz.  │          │
│   │Engine│  │CRUD  │  │Check │  │ Q&A  │  │Tracker   │          │
│   │      │  │Agent │  │      │  │      │  │          │          │
│   └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───────┘          │
│      │         │         │         │          │                    │
│   ML-primary  Entity   ML-assisted FAISS     Rule-based           │
│   + rule      extract  + rule     + MiniLM   + Delta              │
│   fallback    + Delta  fallback   + LLM      queries              │
│               CRUD                grounding                        │
└──────┼─────────┼─────────┼─────────┼──────────┼─────────────────────┘
       │         │         │         │          │
       ▼         ▼         ▼         ▼          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     DATA LAYER (Databricks Lakehouse)                │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Unity Catalog: workspace.asha_copilot                      │   │
│   │                                                             │   │
│   │  Delta Tables:                                              │   │
│   │  ├── patients          (demographics, BPL, Aadhaar)         │   │
│   │  ├── visits            (vitals, symptoms, visit date)       │   │
│   │  ├── immunizations     (vaccine records, due dates)         │   │
│   │  ├── triage_alerts     (risk level, doctor assignment)      │   │
│   │  ├── doctors           (specialization, phone, PHC)         │   │
│   │  ├── phc_facilities    (name, type, district, location)     │   │
│   │  └── scheme_applications (PMMVY/JSY/JSSK status)           │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  FAISS Vector Index (CPU, in-memory, persisted to DBFS)     │   │
│   │  384-dim embeddings from all-MiniLM-L6-v2                   │   │
│   │  Source: NHM/SUMAN clinical protocol PDFs                   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Databricks Secret Scope: "asha-ai"                         │   │
│   │  Keys: sarvam-api-key, hf-token, groq-api-key              │   │
│   └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     ANALYTICS LAYER                                   │
│                                                                      │
│   Supervisor Dashboard (Gradio + Plotly)                             │
│   ├── Risk Distribution (RED/YELLOW/GREEN pie chart)                 │
│   ├── Village-Level Heatmap (stacked bar chart)                      │
│   ├── Immunization Coverage (by vaccine type)                        │
│   ├── High-Risk Patient Table (with doctor assignments)              │
│   └── ASHA Activity Metrics (KPI cards)                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow — Voice Interaction Lifecycle

```
1. ASHA speaks Hindi into mobile browser
       │
2. Web Audio API captures 16 kHz mono WAV → base64
       │
3. Sarvam Saarika ASR → Hindi text
       │
4. Sarvam Mayura Translator → English text
       │
5. LLM Intent Classification:
       ├── "triage"     → Triage Engine (ML + rules)
       ├── "add_patient" → Voice CRUD Agent (entity extraction → Delta INSERT)
       ├── "log_visit"   → Voice CRUD Agent (entity extraction → Delta INSERT)
       ├── "scheme"      → Scheme Eligibility Engine (ML + rules)
       ├── "protocol"    → RAG Pipeline (FAISS retrieval → LLM grounding)
       └── "immunization" → Immunization Tracker (Delta queries)
       │
6. Response generated in English
       │
7. Sarvam Mayura Translator → Hindi text
       │
8. Sarvam Bulbul TTS → Hindi audio
       │
9. Audio + text response returned to ASHA's mobile
```

---

## Component Integration Map

| Component | Source File | Depends On | Writes To |
|-----------|-----------|------------|-----------|
| Voice Pipeline | `src/voice_pipeline.py` | ASR, Translator, LLM, TTS, all engines | Orchestration only |
| Triage Engine | `src/triage_engine.py` | LLM, `danger_signs.yaml`, `phc_doctors.yaml` | `triage_alerts` Delta table |
| Voice CRUD Agent | `src/voice_db_agent.py` | LLM, Translator, Spark | `patients`, `visits` Delta tables |
| Scheme Engine | `src/scheme_eligibility.py` | LLM | `scheme_applications` Delta table |
| RAG Pipeline | `src/rag_pipeline.py` | LLM, sentence-transformers, FAISS | FAISS index (read/write) |
| Dashboard Helpers | `src/dashboard_helpers.py` | Spark | Read-only queries |
| Indic LLM | `src/models/indic_llm.py` | Groq API / HuggingFace | None |
| Indic Translator | `src/models/indic_translate.py` | Sarvam Mayura API / HF | None |
| Indic ASR | `src/models/indic_asr.py` | Sarvam Saarika API / HF | None |
| Indic TTS | `src/models/indic_tts.py` | Sarvam Bulbul API / gTTS | None |

---

## Deployment Modes

| Mode | Entrypoint | Infrastructure |
|------|-----------|----------------|
| **Databricks Apps** | `app/main.py` via `app.yaml` | Serverless container, auto-scaling |
| **In-Notebook** | `notebooks/08_copilot_app.py` | Databricks cluster (interactive) |
| **Dashboard** | `notebooks/09_supervisor_dashboard.py` | Databricks cluster (interactive) |

---

## Design Principles

1. **Voice-First**: Every feature is accessible via Hindi speech — no typing required
2. **Safety-First Triage**: Hard-coded red flags override ML; confidence gating escalates uncertainty to RED
3. **CPU-Only**: No GPU required — all inference via API or quantized CPU models (~85 MB disk, ~250 MB RAM)
4. **Offline-Capable**: FAISS index and embeddings cached locally; graceful degradation when APIs unavailable
5. **Data Sovereignty**: All patient data stays in Databricks workspace (India-hosted, NITI Aayog compliant)
