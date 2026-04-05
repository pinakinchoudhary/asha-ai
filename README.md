# Asha AI

**AI-Powered Maternal & Child Healthcare Assistant for India's Community Health Workers**

Asha AI is a voice-first, multilingual field assistant built on the Databricks Lakehouse platform. It supports India's 1 million+ Accredited Social Health Activists (ASHAs) with clinical triage, NHM protocol guidance, welfare scheme discovery, immunization tracking, and voice-driven patient record management — all running on CPU with open-source Indic AI models and Sarvam AI APIs.

> *"If your solution only works on an A100, it does not work in India."*

---

## Project Write-Up

> Asha AI is a voice-first, multilingual AI copilot on Databricks that empowers India's 1M+ ASHA community health workers with clinical triage (ML + safety rules), NHM protocol RAG, welfare scheme eligibility (PMMVY/JSY/JSSK), immunization tracking, and voice-driven patient CRUD — all via Hindi speech, running on CPU with Delta Lake, FAISS, Sarvam AI APIs, and open-source Indic models, designed for offline-capable, low-resource field deployment.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        ASHA Worker (Mobile)                         │
│                     Hindi / English Voice Input                      │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                               │
│   Gradio UI (Databricks Apps / In-Notebook)                         │
│   ├── Voice Copilot       ├── Triage        ├── Protocol Q&A       │
│   ├── Patient CRUD        ├── Scheme Checker └── Supervisor Dashboard│
│   Web Audio API (16 kHz mono WAV, browser-side JS recording)        │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     SPEECH & LANGUAGE LAYER                           │
│   ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│   │ Sarvam Saarika   │  │ Sarvam Mayura    │  │ Sarvam Bulbul   │   │
│   │ ASR (STT)        │  │ Translation      │  │ TTS             │   │
│   │ Hindi Speech →   │  │ Hindi ↔ English  │  │ Text → Hindi    │   │
│   │ Hindi Text       │  │ Bidirectional    │  │ Audio           │   │
│   └─────────────────┘  └──────────────────┘  └─────────────────┘   │
│   Fallbacks: HF IndicWhisper | HF IndicTrans2 | gTTS               │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     INTELLIGENCE LAYER                                │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  LLM: Groq Llama 3.3 70B / Sarvam-m (30B multilingual)    │   │
│   │  Intent Classification → Route to Appropriate Engine        │   │
│   └──┬──────────┬──────────┬──────────┬──────────┬──────────┘      │
│   ┌──▼───┐  ┌──▼───┐  ┌──▼───┐  ┌──▼───┐  ┌──▼────────┐         │
│   │Triage│  │Voice │  │Scheme│  │ RAG  │  │Immuniz.  │         │
│   │Engine│  │CRUD  │  │Check │  │ Q&A  │  │Tracker   │         │
│   │ML +  │  │Entity│  │ML +  │  │FAISS │  │Rule-based│         │
│   │Rules │  │Extr. │  │Rules │  │+LLM  │  │+ Delta   │         │
│   └──────┘  └──────┘  └──────┘  └──────┘  └──────────┘         │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     DATA LAYER (Databricks Lakehouse)                │
│                                                                      │
│   Unity Catalog: workspace.asha_copilot                              │
│   ┌────────────┬────────────┬──────────────┬──────────────────┐     │
│   │ patients   │ visits     │ immunizations│ triage_alerts    │     │
│   │ doctors    │ phc_facil. │ scheme_apps  │                  │     │
│   └────────────┴────────────┴──────────────┴──────────────────┘     │
│   Delta Lake (ACID, time-travel) │ FAISS Vector Index (CPU)         │
│   Databricks Secret Scopes      │ DBFS (protocol PDFs)             │
└──────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     ANALYTICS LAYER                                   │
│   Supervisor Dashboard (Gradio + Plotly)                             │
│   ├── Risk Distribution (pie)    ├── Village Heatmap (stacked bar)  │
│   ├── Immunization Coverage      ├── High-Risk Patient Table        │
│   └── ASHA Activity Metrics (KPI cards)                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Databricks Technologies Used

| Technology | Usage |
|-----------|-------|
| **Delta Lake** | ACID-compliant storage for all patient, visit, triage, immunization, and scheme data (7 tables) |
| **Unity Catalog** | Data governance — `workspace.asha_copilot` managed schema |
| **Databricks SQL** | Spark SQL analytical queries powering the supervisor dashboard |
| **Databricks Apps** | Production deployment of Gradio UI via `app.yaml` manifest |
| **Databricks Repos** | Git-integrated notebooks synced from GitHub |
| **Databricks Secret Scopes** | Secure API key management (scope: `asha-ai`) |
| **Databricks Connect** | Remote Spark session access (`databricks-connect>=15.4.0`) |
| **DBFS** | Persistent storage for NHM protocol PDFs and FAISS index |
| **Serverless Compute** | On-demand Spark sessions for notebook and app execution |

## Open-Source Models & APIs

| Component | Technology | Notes |
|-----------|-----------|-------|
| **Primary LLM** | Sarvam-m (30B) via Sarvam AI API | Multilingual, strong Hindi, no local GPU |
| **LLM Fallback** | Llama 3.3 70B via Groq API | High-throughput fallback |
| **Translation** | Sarvam Mayura API (fallback: IndicTrans2) | Hindi ↔ English |
| **ASR** | Sarvam Saarika API (fallback: IndicWhisper) | Hindi speech-to-text |
| **TTS** | Sarvam Bulbul v2 API (fallback: gTTS) | Natural Hindi voice |
| **Embeddings** | all-MiniLM-L6-v2 (sentence-transformers) | 384-dim, CPU-friendly, ~80 MB |
| **Vector Search** | FAISS (CPU) | In-memory, persisted to DBFS |
| **UI** | Gradio 4.44 + Plotly | Tabbed interface + interactive charts |

### Resource Budget

| Component | Disk | RAM | GPU |
|-----------|------|-----|-----|
| Sarvam AI APIs (LLM + Translation + ASR + TTS) | 0 GB | 0 GB | None |
| all-MiniLM-L6-v2 | ~80 MB | ~200 MB | None |
| FAISS index | ~5 MB | ~50 MB | None |
| **Total** | **~85 MB** | **~250 MB** | **None** |

---

## Key Features

| Feature | Description | ML Model |
|---------|-------------|----------|
| **Voice-Driven CRUD** | ASHA speaks Hindi to add patients, log visits, update records | Sarvam-m (entity extraction) |
| **Clinical Triage** | ML-primary risk classification (RED/YELLOW/GREEN) with doctor & PHC auto-assignment | Sarvam-m + rule-based fallback |
| **Protocol RAG** | Q&A grounded in NHM/SUMAN clinical guidelines | FAISS + MiniLM-L6-v2 + Sarvam-m |
| **Scheme Eligibility** | PMMVY, JSY, JSSK evaluation with missing document tracking | ML-assisted + rule fallback |
| **Immunization Tracker** | UIP schedule monitoring, overdue flagging, dropout analysis | Rule-based with Delta queries |
| **Supervisor Dashboard** | Real-time risk heatmaps, coverage charts, ASHA activity metrics | Plotly + Gradio |

---

## How to Run

### Prerequisites

- A Databricks account ([Free Edition](https://www.databricks.com/try-databricks) works)
- A GitHub account
- **Sarvam AI API key** — get one at [sarvam.ai](https://www.sarvam.ai) (free tier available)
- Optional: HuggingFace token (for fallback ASR + LLM)

### Step 1: Clone into Databricks

```
Databricks Workspace → Repos → Add Repo
Git URL: https://github.com/pinakinchoudhary/asha-ai.git
→ Click "Create Repo"
```

### Step 2: Set Environment Variables

In **Compute → Edit → Advanced → Environment variables**:

```bash
SARVAM_API_KEY=your_sarvam_key_here    # Required
HF_TOKEN=hf_your_token_here            # Optional fallback
```

### Step 3: Upload NHM Protocol PDFs (Optional, for RAG)

```python
import urllib.request, os
os.makedirs("/dbfs/FileStore/asha_copilot/protocols", exist_ok=True)
urls = {"ASHA_Handbook_HBNC_HBYC.pdf": "https://nhsrcindia.org/sites/default/files/2022-02/Handbook%20for%20ASHA%20Facilitators%20and%20MPWs%20on%20HBNC%20and%20HBYC.pdf"}
for fname, url in urls.items():
    path = f"/dbfs/FileStore/asha_copilot/protocols/{fname}"
    if not os.path.exists(path): urllib.request.urlretrieve(url, path); print(f"Downloaded: {fname}")
```

### Step 4: Run Notebooks in Order

Navigate to `notebooks/` and run each sequentially:

| # | Notebook | What it does | Time |
|---|----------|-------------|------|
| 00 | `00_setup_database` | Creates `workspace.asha_copilot` schema + 7 Delta tables | ~30s |
| 01 | `01_generate_synthetic_data` | Generates 500 patients, visits, immunizations, doctors, PHCs | ~2 min |
| 02 | `02_install_models` | Installs deps, tests Sarvam API connectivity, loads embedder | ~1 min |
| 03 | `03_build_rag_index` | Ingests PDFs, builds FAISS vector index | ~2 min |
| 04 | `04_clinical_triage_demo` | Runs ML triage on visits, assigns doctors | ~5 min |
| 05 | `05_voice_crud_demo` | Demos voice-driven patient add/update | ~2 min |
| 06 | `06_scheme_eligibility_demo` | Evaluates PMMVY/JSY/JSSK eligibility | ~3 min |
| 07 | `07_immunization_tracker` | Flags overdue vaccines, dropout analysis | ~1 min |
| 08 | `08_copilot_app` | **Main demo** — full Gradio app with all features | Interactive |
| 09 | `09_supervisor_dashboard` | Supervisor analytics dashboard | Interactive |

### Step 5: Launch the Demo

```
Run notebook 08_copilot_app → Gradio app launches in-notebook
Run notebook 09_supervisor_dashboard → Analytics dashboard launches
```

### Running Tests Locally

```bash
pip install pyyaml pytest
pytest tests/ -v
```

---

## Demo Steps

### 1. Voice-Driven Patient Registration (Notebook 08, Tab: Patient CRUD)

```
Click "Record" → Speak: "Naya patient register karo — Sunita Devi, umra 24, gaon Rampur, BPL card hai, Aadhaar hai"
→ Click "Send Voice"
→ System extracts entities, writes to Delta table, confirms in Hindi
```

### 2. Voice-Driven Visit Logging + Auto-Triage (Notebook 08, Tab: Voice Copilot)

```
Click "Record" → Speak: "Sunita Devi ka checkup — BP 155/100, hemoglobin 6.5, tez sir dard aur pair sooje hain"
→ Click "Send Voice"
→ System logs visit → ML triage: RED → Assigns Dr. Meena Tripathi at District Hospital Varanasi
→ Response: "ATYANT ZARURI — Pre-eclampsia + severe anemia. District Hospital Varanasi bhejein."
```

### 3. Scheme Eligibility Check (Notebook 08, Tab: Scheme Checker)

```
Type patient name: "Sunita"
→ Click "Check Eligibility"
→ PMMVY: Eligible (Rs 5,000) | JSY: Eligible (Rs 1,400 — UP is LPS) | JSSK: Eligible (free delivery)
```

### 4. Protocol Q&A (Notebook 08, Tab: Protocol Q&A)

```
Type: "What are the danger signs during pregnancy?"
→ Click "Ask"
→ RAG retrieves from NHM protocol PDFs → Grounded answer with source attribution
```

### 5. Supervisor Dashboard (Notebook 09)

```
Run notebook → Dashboard auto-loads with:
→ KPI cards: Total patients, RED alerts, Overdue vaccines
→ Risk pie chart (RED/YELLOW/GREEN distribution)
→ Village-level stacked bar chart
→ Immunization coverage by vaccine
→ High-risk patient table with assigned doctors
```

---

## Project Structure

```
asha-ai/
├── README.md
├── requirements.txt
├── app.yaml                         # Databricks Apps deployment manifest
├── config/
│   ├── settings.py                  # Central configuration
│   ├── danger_signs.yaml            # Clinical triage rules (SUMAN/NHM)
│   └── phc_doctors.yaml             # PHC facilities + doctor registry
├── data/nhm_protocols/              # Upload NHM PDFs here
├── docs/                            # Detailed use case documentation
│   ├── 01_architecture_overview.md
│   ├── 02_tech_stack.md
│   ├── 03_voice_patient_management.md
│   ├── 04_clinical_triage.md
│   ├── 05_protocol_rag.md
│   ├── 06_scheme_eligibility.md
│   ├── 07_immunization_tracker.md
│   └── 08_supervisor_dashboard.md
├── src/
│   ├── models/
│   │   ├── indic_llm.py             # LLM wrapper (Groq/Sarvam/HF)
│   │   ├── indic_translate.py       # Translation (Sarvam Mayura/HF)
│   │   ├── indic_asr.py             # ASR (Sarvam Saarika/HF)
│   │   └── indic_tts.py             # TTS (Sarvam Bulbul/gTTS)
│   ├── triage_engine.py             # ML triage + doctor/PHC assignment
│   ├── scheme_eligibility.py        # PMMVY/JSY/JSSK eligibility
│   ├── rag_pipeline.py              # FAISS RAG over NHM protocols
│   ├── voice_db_agent.py            # Voice-driven CRUD agent
│   ├── voice_pipeline.py            # End-to-end voice orchestration
│   └── dashboard_helpers.py         # Dashboard query utilities
├── notebooks/                       # Databricks notebooks (run in order)
│   ├── 00_setup_database.py
│   ├── 01_generate_synthetic_data.py
│   ├── 02_install_models.py
│   ├── 03_build_rag_index.py
│   ├── 04_clinical_triage_demo.py
│   ├── 05_voice_crud_demo.py
│   ├── 06_scheme_eligibility_demo.py
│   ├── 07_immunization_tracker.py
│   ├── 08_copilot_app.py            # Main demo app
│   └── 09_supervisor_dashboard.py
├── app/
│   └── main.py                      # Databricks App entrypoint
└── tests/
    └── test_triage.py               # pytest (runs locally)
```

---

## Clinical Safety

- **Hard-coded red flags** (convulsions, severe bleeding, chest indrawing, foul lochia, yellow palms/soles, unable to feed) **always trigger emergency referral** — no ML can override these
- **Confidence gating**: If ML triage confidence < 85%, the system automatically escalates to RED with "consult a doctor immediately"
- **Rule-based fallback**: If all ML models fail, deterministic rules from SUMAN/NHM guidelines take over
- **No hallucination**: RAG answers are grounded in official NHM protocol PDFs only

---

## Data Sources

| Source | Use |
|--------|-----|
| [NHM HBNC/HBYC Handbook](https://nhsrcindia.org/) | RAG knowledge base |
| [SUMAN Guidelines](https://www.ncbi.nlm.nih.gov/books/NBK304178/) | Danger sign rules |
| [NFHS-5 Factsheets](https://dhsprogram.com/pubs/pdf/FR375/FR375.pdf) | Synthetic data calibration |
| [UIP Schedule (WHO India)](https://cdn.who.int/media/docs/default-source/searo/india/publications/immunization-handbook-1-106-part1.pdf) | Immunization tracking |
| [PMMVY](https://pmmvy.wcd.gov.in/) / [JSY](https://www.myscheme.gov.in/schemes/jsy1) / [JSSK](https://www.myscheme.gov.in/schemes/jssk) | Scheme eligibility rules |

---

## Detailed Documentation

See the [docs/](./docs/) folder for in-depth documentation on each use case:

- [Architecture Overview](./docs/01_architecture_overview.md) — Full system architecture with component diagrams
- [Tech Stack](./docs/02_tech_stack.md) — Databricks technologies, open-source models, APIs
- [Voice Patient Management](./docs/03_voice_patient_management.md) — Voice CRUD data flow and design
- [Clinical Triage](./docs/04_clinical_triage.md) — ML-primary triage with safety-first fallback
- [Protocol RAG](./docs/05_protocol_rag.md) — NHM protocol Q&A via FAISS + LLM
- [Scheme Eligibility](./docs/06_scheme_eligibility.md) — PMMVY/JSY/JSSK eligibility engine
- [Immunization Tracker](./docs/07_immunization_tracker.md) — UIP schedule monitoring and dropout analysis
- [Supervisor Dashboard](./docs/08_supervisor_dashboard.md) — Real-time analytics for supervisors

---

Built for the Databricks Hackathon. Designed for India.
