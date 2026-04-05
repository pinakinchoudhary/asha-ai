# Asha AI — Documentation

**AI-Powered Maternal & Child Healthcare Assistant for India's Community Health Workers**

This folder contains detailed documentation for each use case, the overall architecture, tech stack, and data flow of the Asha AI platform.

---

## Table of Contents

| Document | Description |
|----------|-------------|
| [Architecture Overview](./01_architecture_overview.md) | System architecture, component diagram, and integration patterns |
| [Tech Stack](./02_tech_stack.md) | Databricks-centric tech stack, open-source models, and APIs |
| [Voice-Driven Patient Management](./03_voice_patient_management.md) | Voice CRUD: register patients, log visits via Hindi/English speech |
| [Clinical Triage Engine](./04_clinical_triage.md) | ML-primary risk classification with safety-first fallback design |
| [Protocol RAG (Q&A)](./05_protocol_rag.md) | Retrieval-Augmented Generation over NHM/SUMAN clinical guidelines |
| [Government Scheme Eligibility](./06_scheme_eligibility.md) | PMMVY, JSY, JSSK eligibility evaluation and document tracking |
| [Immunization Tracker](./07_immunization_tracker.md) | UIP schedule monitoring, overdue flagging, dropout analysis |
| [Supervisor Dashboard](./08_supervisor_dashboard.md) | Real-time analytics, risk heatmaps, and coverage monitoring |

---

## Project Write-Up

Asha AI is a voice-first, multilingual AI copilot built on the Databricks Lakehouse platform that empowers India's 1M+ community health workers (ASHAs) with clinical triage, NHM protocol guidance, welfare scheme discovery, immunization tracking, and voice-driven patient management — all running on CPU with open-source Indic AI models and Sarvam APIs, designed for low-resource, offline-capable field deployment.

---

## Quick Links

- **Main App**: `notebooks/08_copilot_app.py` or `app/main.py` (Databricks Apps)
- **Supervisor Dashboard**: `notebooks/09_supervisor_dashboard.py`
- **Configuration**: `config/settings.py`
- **Source Code**: `src/`
