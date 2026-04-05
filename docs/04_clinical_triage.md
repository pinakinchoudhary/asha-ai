# Use Case: Clinical Triage Engine

## Problem Statement

ASHA workers encounter pregnant women and newborns with a wide range of symptoms in the field. They must quickly determine whether a case is an emergency (requiring immediate hospital referral), a warning (requiring doctor consultation within hours), or routine (safe for home-based follow-up). Incorrect classification can be fatal — delayed referrals for pre-eclampsia, severe anemia, or neonatal danger signs are a leading cause of preventable maternal and infant mortality in India.

---

## Solution

Asha AI implements an **ML-primary, safety-first clinical triage engine** that classifies patient risk into three levels — **RED** (emergency), **YELLOW** (warning), **GREEN** (routine) — and automatically assigns the nearest appropriate doctor and Primary Health Centre (PHC). The system uses hard-coded red flags that **always override ML**, ensuring that life-threatening conditions are never downgraded.

---

## Data Flow

```
Patient vitals + symptoms (from visit record or voice input)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  STAGE 1: Hard-Coded Red Flag Check              │
│                                                   │
│  danger_signs.yaml — SUMAN/NHM guidelines         │
│  ├── Convulsions          → RED (immediate)       │
│  ├── Severe bleeding      → RED (immediate)       │
│  ├── Chest indrawing      → RED (immediate)       │
│  ├── Foul lochia          → RED (immediate)       │
│  ├── Yellow palms/soles   → RED (immediate)       │
│  └── Unable to feed       → RED (immediate)       │
│                                                   │
│  If ANY red flag present → SKIP ML → RED          │
└──────────┬────────────────────────────────────────┘
           │ No red flags
           ▼
┌─────────────────────────────────────────────────┐
│  STAGE 2: ML-Based Risk Classification           │
│                                                   │
│  LLM (Sarvam-m / Groq Llama 3.3 70B)            │
│  Input: vitals + symptoms + patient history       │
│  Output: risk_level + confidence + reasoning      │
│                                                   │
│  Confidence < 85% → AUTO-ESCALATE to RED         │
│  "When in doubt, refer."                          │
└──────────┬────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│  STAGE 3: Rule-Based Fallback                    │
│                                                   │
│  If ML fails completely:                          │
│  ├── BP ≥ 160/110 → RED                          │
│  ├── Hemoglobin < 7 → RED                        │
│  ├── BP 140-159/90-109 → YELLOW                  │
│  ├── Hemoglobin 7-9 → YELLOW                     │
│  └── All else → GREEN                            │
└──────────┬────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│  STAGE 4: Doctor & PHC Assignment                │
│                                                   │
│  phc_doctors.yaml — facility registry             │
│  ├── RED → District Hospital + Specialist         │
│  ├── YELLOW → CHC + General Physician             │
│  └── GREEN → Sub-Centre / PHC                     │
│                                                   │
│  Output: doctor_name, phone, PHC, urgency_hours   │
└──────────┬────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│  Delta Lake: triage_alerts table                  │
│  ├── patient_id, risk_level, sign_name            │
│  ├── action, urgency_hours                        │
│  ├── assigned_phc_name, assigned_doctor_name      │
│  └── assigned_doctor_phone, alert_date            │
└───────────────────────────────────────────────────┘
```

---

## Architecture

| Layer | Component | Technology |
|-------|-----------|-----------|
| **Red Flag Detection** | Hard-coded rules | YAML-driven (`config/danger_signs.yaml`) |
| **ML Classification** | LLM-based risk assessment | Sarvam-m / Groq Llama 3.3 70B |
| **Confidence Gating** | Auto-escalation on low confidence | Threshold: 85% |
| **Rule Fallback** | Deterministic vitals-based rules | Python logic from SUMAN/NHM guidelines |
| **Doctor Assignment** | Facility registry matching | YAML-driven (`config/phc_doctors.yaml`) |
| **Storage** | Triage results and audit trail | Delta Lake (`workspace.asha_copilot.triage_alerts`) |

---

## Safety Design

| Mechanism | Purpose | Behavior |
|-----------|---------|----------|
| **Hard-coded red flags** | Prevent ML from missing life-threatening conditions | Always triggers RED — no ML can override |
| **Confidence gating (85%)** | Handle ML uncertainty safely | Auto-escalates to RED with "consult a doctor immediately" |
| **Rule-based fallback** | Graceful degradation when ML/API fails | Deterministic triage from vitals alone |
| **Audit trail** | Accountability and review | Every triage result persisted to Delta Lake with timestamp |

---

## Key Source Files

| File | Purpose |
|------|---------|
| `src/triage_engine.py` | Core triage logic — ML classification, red flag checks, doctor assignment |
| `config/danger_signs.yaml` | SUMAN/NHM-derived clinical red flag rules (123 rules) |
| `config/phc_doctors.yaml` | PHC facility and doctor registry |
| `notebooks/04_clinical_triage_demo.py` | Interactive triage demo with sample patients |
| `tests/test_triage.py` | Unit tests for triage safety guarantees |

---

## Demo

```
Input:  BP 155/100, Hemoglobin 6.5, Symptoms: severe headache, swollen feet
Output: Risk Level: RED
        Diagnosis: Pre-eclampsia + Severe Anemia
        Action: "ATYANT ZARURI — District Hospital Varanasi bhejein"
        Doctor: Dr. Meena Tripathi, Phone: +91-9876543005
        Urgency: Immediate (0 hours)
```
