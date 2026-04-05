# Use Case: Voice-Driven Patient Management

## Problem Statement

India's ASHA workers serve remote, low-literacy populations and operate in the field with limited connectivity. Typing patient records into digital systems is impractical — most ASHAs are comfortable speaking Hindi, not typing structured data into forms. This creates a data entry bottleneck that leads to incomplete patient records, missed follow-ups, and poor data quality at the district and state level.

---

## Solution

Asha AI provides a **voice-first CRUD interface** that allows ASHA workers to register patients, log visits, and update records entirely through Hindi (or English) speech. The system automatically extracts structured entities from natural language and writes them to Delta Lake tables.

---

## Data Flow

```
ASHA speaks Hindi
    │
    ▼
┌──────────────────────────┐
│  Web Audio API            │
│  16 kHz mono WAV capture  │
│  → base64 encoding        │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Sarvam Saarika ASR       │
│  Hindi Speech → Hindi Text│
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Sarvam Mayura Translator │
│  Hindi Text → English Text│
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  LLM Intent Detection     │
│  Classify: add_patient,   │
│  log_visit, update_patient│
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  LLM Entity Extraction    │
│  Extract: name, age,      │
│  village, BPL, vitals,    │
│  symptoms, Aadhaar        │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Delta Lake CRUD          │
│  INSERT INTO patients     │
│  INSERT INTO visits       │
│  UPDATE patients          │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Confirmation Response    │
│  English → Hindi (Mayura) │
│  Hindi Text → Hindi Audio │
│  (Bulbul TTS)             │
└──────────────────────────┘
```

---

## Architecture

| Layer | Component | Technology |
|-------|-----------|-----------|
| **Input** | Voice recording | Web Audio API (browser-side JavaScript) |
| **ASR** | Speech-to-text | Sarvam Saarika API (fallback: HF IndicWhisper) |
| **Translation** | Hindi → English | Sarvam Mayura API (fallback: HF IndicTrans2) |
| **NLU** | Intent + entity extraction | Groq Llama 3.3 70B / Sarvam-m |
| **Storage** | Patient records | Delta Lake (`workspace.asha_copilot.patients`, `.visits`) |
| **Response** | English → Hindi | Sarvam Mayura + Bulbul TTS |

---

## Supported Commands

| Voice Command (Hindi) | Intent | Delta Table Action |
|-----------------------|--------|-------------------|
| "Naya patient register karo — Sunita Devi, umra 24, gaon Rampur" | `add_patient` | `INSERT INTO patients` |
| "Sunita Devi ka checkup — BP 140/90, hemoglobin 8.5" | `log_visit` | `INSERT INTO visits` |
| "Sunita ka Aadhaar update karo — 1234-5678-9012" | `update_patient` | `UPDATE patients SET aadhaar = ...` |

---

## Key Source Files

| File | Purpose |
|------|---------|
| `src/voice_db_agent.py` | Voice command processing, entity extraction, Delta CRUD |
| `src/voice_pipeline.py` | End-to-end voice orchestration (ASR → Translate → Intent → CRUD → TTS) |
| `src/models/indic_asr.py` | Sarvam Saarika ASR wrapper with HuggingFace fallback |
| `src/models/indic_translate.py` | Sarvam Mayura translation wrapper |
| `src/models/indic_llm.py` | LLM wrapper (Groq API + HuggingFace fallback) |
| `notebooks/05_voice_crud_demo.py` | Interactive demo notebook |

---

## Demo

```
Input:  "Naya patient register karo — Sunita Devi, umra 24, gaon Rampur, BPL card hai, Aadhaar hai"
Output: System extracts entities → writes to Delta table → confirms in Hindi:
        "Sunita Devi ka registration ho gaya. Patient ID: P-0501. Gaon: Rampur, Umra: 24, BPL: Haan."
```
