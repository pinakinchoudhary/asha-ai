# ASHA Copilot

**AI-Powered Maternal & Child Healthcare Assistant for India's Community Health Workers**

ASHA Copilot is a voice-first, multilingual field assistant built on the Databricks Lakehouse platform. It supports India's 1 million+ Accredited Social Health Activists (ASHAs) with clinical triage, NHM protocol guidance, welfare scheme discovery, immunization tracking, and voice-driven patient record management — all running on CPU with quantized open-source Indic AI models.

> *"If your solution only works on an A100, it does not work in India."*

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │     ASHA Worker (Mobile)     │
                        │   Hindi / English Voice      │
                        └──────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   IndicWhisper ASR (API)     │
                    │   Speech → Hindi Text        │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   IndicTrans2 (ONNX/API)     │
                    │   Hindi → English             │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │         Intent Classification            │
              │    Airavata / Param-1 (GGUF on CPU)     │
              └──┬──────┬──────┬──────┬──────┬─────────┘
                 │      │      │      │      │
         ┌───▼──┐ ┌──▼───┐ ┌──▼──┐ ┌──▼──┐ ┌─▼────┐
         │Triage│ │ CRUD │ │Scheme│ │ RAG │ │ Imm. │
         │Engine│ │Agent │ │Check │ │ Q&A │ │Track │
         └──┬───┘ └──┬───┘ └──┬──┘ └──┬──┘ └──┬───┘
            │        │        │       │        │
            └────────┴────────┴───┬───┴────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  Delta Lake (hive_metastore) │
                    │  patients | visits | doctors │
                    │  immunizations | triage_alerts│
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   IndicTrans2 (en → hi)    │
                    │   gTTS / Sarvam TTS        │
                    │   Response → Hindi Audio    │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   Supervisor Dashboard      │
                    │   Gradio (Plotly charts)    │
                    └─────────────────────────────┘
```

## Key Features

| Feature | Description | ML Model |
|---------|-------------|----------|
| **Voice-Driven CRUD** | ASHA speaks in Hindi to add patients, log visits, update records | Sarvam-m (entity extraction) |
| **Clinical Triage** | ML-primary risk classification (RED/YELLOW/GREEN) with doctor & PHC auto-assignment | Sarvam-m + rule-based fallback |
| **Protocol RAG** | Q&A grounded in NHM/SUMAN clinical guidelines | FAISS + MiniLM-L6-v2 + Sarvam-m |
| **Scheme Eligibility** | PMMVY, JSY, JSSK evaluation with missing document tracking | ML-assisted + rule fallback |
| **Immunization Tracker** | UIP schedule monitoring, overdue flagging, dropout analysis | Rule-based with Delta queries |
| **Supervisor Dashboard** | Real-time risk heatmaps, coverage charts, ASHA activity metrics | Plotly + Gradio |

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| **Platform** | Databricks (workspace catalog) | Serverless-compatible |
| **Storage** | Delta Lake (workspace.asha_copilot) | Unity Catalog |
| **Primary LLM** | sarvam-30b via Sarvam AI API | 30B multilingual model, strong Hindi, no local GPU needed |
| **Translation** | Sarvam Mayura API | Hindi ↔ English |
| **ASR** | Sarvam Saarika API | Hindi speech-to-text |
| **TTS** | Sarvam Bulbul v2 API | Natural Hindi voice |
| **Embeddings** | all-MiniLM-L6-v2 (auto-download ~80 MB) | 384-dim, CPU-friendly |
| **Vector Search** | FAISS (CPU) | In-memory, persisted to Workspace |
| **UI** | Gradio | In-notebook, no separate server |

### Resource Budget

| Component | Local Disk | RAM | Purpose |
|-----------|-----------|-----|---------|
| Sarvam AI API (LLM + Translation + TTS + ASR) | 0 GB | 0 GB | All AI inference via API |
| all-MiniLM-L6-v2 | ~80 MB | ~200 MB | RAG embeddings only |
| FAISS index | ~5 MB | ~50 MB | Vector search over NHM protocols |
| **Total** | **~85 MB** | **~250 MB** | Minimal local footprint |

---

## Databricks Setup Guide

### Prerequisites

- A Databricks account (Free Edition works)
- A GitHub account (to clone this repo)
- **Sarvam AI API key** — get one at [sarvam.ai](https://www.sarvam.ai) (free tier available)
- Optional: Hugging Face token (for ASR + LLM fallback)

### Step 1: Create Databricks Workspace

1. Go to [Databricks Free Edition](https://www.databricks.com/try-databricks) and sign up
2. Select your cloud provider and create a workspace
3. Wait for the workspace to be provisioned

### Step 2: Clone Repository into Databricks

1. In Databricks, go to **Workspace** > **Repos** (left sidebar)
2. Click **Add Repo**
3. Enter the Git URL: `https://github.com/pinakinchoudhary/asha-ai.git`
4. Click **Create Repo**
5. The entire project structure will be available under `/Workspace/Repos/<your-username>/asha-ai/`

### Step 3: Upload NHM Protocol PDFs (Optional, for RAG)

Download any of these NHM protocol PDFs and upload to DBFS:

- [Handbook for ASHA Facilitators (HBNC/HBYC)](https://nhsrcindia.org/sites/default/files/2022-02/Handbook%20for%20ASHA%20Facilitators%20and%20MPWs%20on%20HBNC%20and%20HBYC.pdf)
- [ASHA Module 6: Skills that Save Lives](https://nrhmmanipur.org/wp-content/uploads/2011/01/ASHA-Module-6.pdf)
- [Routine Immunization Handbook](https://cdn.who.int/media/docs/default-source/searo/india/publications/immunization-handbook-1-106-part1.pdf)

Upload via a notebook cell:
```python
# Upload from your local machine to DBFS
import urllib.request
import os

os.makedirs("/dbfs/FileStore/asha_copilot/protocols", exist_ok=True)

urls = {
    "ASHA_Handbook_HBNC_HBYC.pdf": "https://nhsrcindia.org/sites/default/files/2022-02/Handbook%20for%20ASHA%20Facilitators%20and%20MPWs%20on%20HBNC%20and%20HBYC.pdf",
}
for fname, url in urls.items():
    path = f"/dbfs/FileStore/asha_copilot/protocols/{fname}"
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded: {fname}")
```

### Step 4: Set Environment Variables (Optional)

Set these in your Databricks cluster **Environment Variables** (Compute → Edit → Advanced → Environment variables) for persistence across restarts:

```
SARVAM_API_KEY=your_sarvam_key_here    # Required — get at sarvam.ai
HF_TOKEN=hf_your_token_here            # Optional — HF API fallback for ASR
```

Or set them at the top of notebook `02_install_models` for a single session.

### Step 5: Run Notebooks in Order

Navigate to the `notebooks/` folder in your cloned repo and run each notebook sequentially:

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

### Step 6: Run the Demo

1. Run notebook **08_copilot_app** — this launches the interactive Gradio app
2. Use the **Voice Copilot** tab to chat in Hindi/English
3. Use the **Triage** tab to input vitals and see risk classification + doctor assignment
4. Use the **Patient CRUD** tab to add patients via natural language
5. Run notebook **09_supervisor_dashboard** for the analytics overview

---

## Demo Script (for Hackathon Judges)

### 1. Voice-Driven Patient Registration (Notebook 08, Tab: Patient CRUD)
```
Input: "Naya patient register karo — Sunita Devi, umra 24, gaon Rampur, BPL card hai, Aadhaar hai"
→ System extracts entities, writes to Delta table, confirms in Hindi
```

### 2. Voice-Driven Visit Logging + Auto-Triage (Notebook 08, Tab: Voice Copilot)
```
Input: "Sunita Devi ka checkup — BP 155/100, hemoglobin 6.5, tez sir dard aur pair sooje hain"
→ System logs visit → ML triage: RED → Assigns Dr. Meena Tripathi at District Hospital Varanasi
→ Response: "ATYANT ZARURI — Pre-eclampsia + severe anemia. District Hospital Varanasi bhejein. Dr. Meena Tripathi, Phone: +91-9876543005"
```

### 3. Scheme Eligibility Check (Notebook 08, Tab: Scheme Checker)
```
Input: Patient name "Sunita"
→ PMMVY: Eligible (Rs 5,000) | JSY: Eligible (Rs 1,400 — UP is LPS) | JSSK: Eligible (free delivery)
```

### 4. Protocol Q&A (Notebook 08, Tab: Protocol Q&A)
```
Input: "What are the danger signs during pregnancy?"
→ RAG retrieves from NHM protocol PDFs → Grounded answer with sources
```

### 5. Supervisor Dashboard (Notebook 09)
```
→ KPI cards: Total patients, Red alerts, Overdue vaccines
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
├── .gitignore
├── config/
│   ├── settings.py              # Central configuration
│   ├── danger_signs.yaml        # Clinical triage rules (SUMAN/NHM)
│   └── phc_doctors.yaml         # PHC facilities + doctor registry
├── data/nhm_protocols/          # Upload NHM PDFs here
├── src/
│   ├── models/
│   │   ├── indic_llm.py         # Airavata/Param-1 GGUF wrapper
│   │   ├── indic_translate.py   # IndicTrans2 translator
│   │   ├── indic_asr.py         # IndicWhisper ASR
│   │   └── indic_tts.py         # TTS (Sarvam/gTTS)
│   ├── synthetic_data.py        # NFHS-5 calibrated data generators
│   ├── triage_engine.py         # ML triage + doctor/PHC assignment
│   ├── scheme_eligibility.py    # PMMVY/JSY/JSSK eligibility
│   ├── rag_pipeline.py          # FAISS RAG over NHM protocols
│   ├── voice_db_agent.py        # Voice-driven CRUD agent
│   ├── voice_pipeline.py        # End-to-end voice orchestration
│   └── dashboard_helpers.py     # Dashboard query utilities
├── notebooks/                   # Databricks notebooks (run in order)
│   ├── 00_setup_database.py
│   ├── 01_generate_synthetic_data.py
│   ├── 02_install_models.py
│   ├── 03_build_rag_index.py
│   ├── 04_clinical_triage_demo.py
│   ├── 05_voice_crud_demo.py
│   ├── 06_scheme_eligibility_demo.py
│   ├── 07_immunization_tracker.py
│   ├── 08_copilot_app.py        # Main demo app
│   └── 09_supervisor_dashboard.py
└── tests/
    └── test_triage.py           # pytest (runs locally)
```

## Running Tests Locally

```bash
pip install pyyaml pytest
pytest tests/ -v
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

Built for the Databricks Hackathon. Designed for India.
