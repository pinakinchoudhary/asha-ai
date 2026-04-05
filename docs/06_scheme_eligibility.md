# Use Case: Government Welfare Scheme Eligibility

## Problem Statement

India's maternal and child health welfare schemes — PMMVY, JSY, JSSK — provide critical financial support and free services to pregnant women and newborns. However, ASHA workers often struggle to determine which patients qualify for which schemes, what documents are missing, and what the entitlements are. This leads to under-enrollment, delayed benefits, and missed financial support for the families who need it most.

---

## Solution

Asha AI implements an **ML-assisted scheme eligibility engine** that evaluates patient records against the official rules of three major government welfare schemes. It identifies eligibility, calculates entitlements, tracks missing documents, and provides actionable next steps — all accessible via voice or the Gradio UI.

---

## Supported Schemes

### PMMVY (Pradhan Mantri Matru Vandana Yojana)

| Parameter | Details |
|-----------|---------|
| **Benefit** | Rs 5,000–6,000 cash transfer for first live birth |
| **Eligibility** | First pregnancy, age ≥ 19, registered at AWC/health facility |
| **Documents Required** | Aadhaar, bank account, MCP card, pregnancy registration |
| **Exclusions** | Government employees, women already receiving maternity benefits |

### JSY (Janani Suraksha Yojana)

| Parameter | Details |
|-----------|---------|
| **Benefit** | Rs 700 (HPS) to Rs 1,400 (LPS) for institutional delivery |
| **Eligibility** | BPL/SC/ST in LPS states; age ≥ 19 in HPS states |
| **LPS States** | UP, Bihar, MP, Rajasthan, Jharkhand, Chhattisgarh, Odisha, Uttarakhand, J&K, Assam |
| **Documents Required** | BPL card, JSY card, institutional delivery certificate |

### JSSK (Janani Shishu Suraksha Karyakram)

| Parameter | Details |
|-----------|---------|
| **Benefit** | Free delivery, C-section, drugs, diagnostics, blood, transport, diet |
| **Eligibility** | All pregnant women delivering in public health institutions |
| **Coverage** | Mother (up to 48 hours postpartum) + sick newborn (up to 30 days) |

---

## Data Flow

```
Patient record (from Delta Lake or voice input)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  STAGE 1: ML-Assisted Evaluation                  │
│                                                   │
│  LLM (Sarvam-m / Groq Llama 3.3 70B)            │
│  Input: patient demographics + medical history    │
│  Output: per-scheme eligibility + reasoning       │
│                                                   │
│  If ML succeeds → parse structured results        │
└──────────┬────────────────────────────────────────┘
           │ ML fails or returns invalid
           ▼
┌──────────────────────────────────────────────────┐
│  STAGE 2: Rule-Based Fallback                     │
│                                                   │
│  Deterministic rules per scheme:                  │
│  ├── PMMVY: first_pregnancy AND age ≥ 19          │
│  ├── JSY: bpl_status AND state in LPS_STATES      │
│  └── JSSK: institutional delivery (always elig.)  │
└──────────┬────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Output per scheme:                               │
│  ├── eligible: true/false                         │
│  ├── amount: Rs value                             │
│  ├── missing_requirements: [list]                 │
│  ├── entitlements: [list of benefits]             │
│  └── notes: actionable next steps                 │
└──────────┬────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Delta Lake: scheme_applications table            │
│  Tracks enrollment status per patient per scheme  │
└──────────────────────────────────────────────────┘
```

---

## Architecture

| Layer | Component | Technology |
|-------|-----------|-----------|
| **ML Evaluation** | LLM-based eligibility assessment | Sarvam-m / Groq Llama 3.3 70B |
| **Rule Fallback** | Deterministic eligibility rules | Python logic from official scheme guidelines |
| **State Classification** | LPS vs HPS state detection (for JSY) | Hard-coded `LPS_STATES` set |
| **Storage** | Scheme application tracking | Delta Lake (`workspace.asha_copilot.scheme_applications`) |

---

## Key Source Files

| File | Purpose |
|------|---------|
| `src/scheme_eligibility.py` | ML-assisted eligibility engine with rule-based fallback |
| `notebooks/06_scheme_eligibility_demo.py` | Interactive demo with sample patients |

---

## Demo

```
Input:  Patient "Sunita Devi" — age 24, first pregnancy, BPL, UP, Aadhaar: yes

Output:
  PMMVY: ✅ Eligible — Rs 5,000 (first live birth)
         Missing: MCP card registration
         
  JSY:   ✅ Eligible — Rs 1,400 (UP is Low Performing State)
         Missing: JSY card
         
  JSSK:  ✅ Eligible — Free delivery + transport + drugs + diagnostics
         No missing documents
```
