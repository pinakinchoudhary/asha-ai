# Use Case: Immunization Tracker

## Problem Statement

India's Universal Immunization Programme (UIP) mandates a complex multi-dose vaccine schedule for children from birth to 16 months. ASHA workers are responsible for tracking which children have received which vaccines, identifying overdue doses, and ensuring dropout rates stay low. With hundreds of children per ASHA, manual tracking on paper registers is error-prone, leading to missed vaccinations and outbreaks of preventable diseases.

---

## Solution

Asha AI implements a **rule-based immunization tracking system** backed by Delta Lake that monitors each child's vaccine records against the UIP schedule, flags overdue doses, calculates dropout rates by village, and surfaces actionable follow-up lists for ASHA workers and supervisors.

---

## UIP Vaccine Schedule

| Vaccine | Dose | Due Age | Route |
|---------|------|---------|-------|
| BCG | 1 | Birth | Intradermal |
| OPV-0 | 0 (birth) | Birth | Oral |
| Hepatitis B | Birth | Birth | IM |
| OPV | 1, 2, 3 | 6, 10, 14 weeks | Oral |
| Pentavalent (DPT+HepB+Hib) | 1, 2, 3 | 6, 10, 14 weeks | IM |
| Rotavirus | 1, 2, 3 | 6, 10, 14 weeks | Oral |
| IPV | 1, 2 | 6, 14 weeks | IM |
| Measles/MR | 1 | 9 months | SC |
| Vitamin A | 1 | 9 months | Oral |
| JE (endemic areas) | 1 | 9 months | SC |
| DPT Booster | 1 | 16–24 months | IM |
| Measles/MR | 2 | 16–24 months | SC |
| OPV Booster | 1 | 16–24 months | Oral |

---

## Data Flow

```
Delta Lake: immunizations table
(patient_id, vaccine_name, dose_number, date_given, date_due)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Schedule Comparison                              │
│                                                   │
│  For each child:                                  │
│  ├── Expected vaccines by age (from UIP schedule) │
│  ├── Received vaccines (from Delta records)       │
│  └── Gap = Expected − Received                    │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Overdue Flagging                                 │
│                                                   │
│  date_due < today AND date_given IS NULL          │
│  → Flag as OVERDUE                                │
│  → Priority: days overdue                         │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Dropout Analysis                                 │
│                                                   │
│  Per village, per vaccine:                        │
│  ├── Started (dose 1 given)                       │
│  ├── Completed (all doses given)                  │
│  └── Dropout Rate = (Started − Completed)/Started │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Outputs:                                         │
│  ├── Overdue vaccine list per child               │
│  ├── Village-level coverage charts                │
│  ├── Dropout analysis by vaccine type             │
│  └── Follow-up priority list for ASHA             │
└──────────────────────────────────────────────────┘
```

---

## Architecture

| Layer | Component | Technology |
|-------|-----------|-----------|
| **Data Source** | Vaccine administration records | Delta Lake (`workspace.asha_copilot.immunizations`) |
| **Schedule Rules** | UIP vaccine schedule | Hard-coded Python rules |
| **Query Engine** | Overdue detection and dropout analysis | Databricks SQL (Spark SQL over Delta) |
| **Visualization** | Coverage charts and dropout analysis | Plotly (bar charts, line charts) |
| **Dashboard** | Supervisor-level immunization metrics | Gradio + Plotly in supervisor dashboard |

---

## Key Metrics

| Metric | Description | Query Source |
|--------|-------------|-------------|
| **Overdue Count** | Children with vaccines past due date | `immunizations WHERE date_due < today AND date_given IS NULL` |
| **Coverage Rate** | % of children fully vaccinated by age group | `immunizations GROUP BY vaccine_name` |
| **Dropout Rate** | % who started but didn't complete a vaccine series | `(dose_1_count - final_dose_count) / dose_1_count` |
| **Village Coverage** | Immunization coverage per village | `JOIN patients ON patient_id GROUP BY village` |

---

## Key Source Files

| File | Purpose |
|------|---------|
| `notebooks/07_immunization_tracker.py` | Overdue flagging, dropout analysis, coverage charts |
| `src/dashboard_helpers.py` | SQL queries for immunization metrics |
| `notebooks/00_setup_database.py` | Creates the `immunizations` Delta table |
| `notebooks/01_generate_synthetic_data.py` | Generates synthetic immunization records |

---

## Demo

```
Output:
  Overdue Vaccines:
  ├── Ravi Kumar (age 8 months, village Rampur) — Pentavalent Dose 3 overdue by 14 days
  ├── Priya Singh (age 11 months, village Sultanpur) — Measles/MR Dose 1 overdue by 21 days
  └── Anil Yadav (age 6 months, village Jaunpur) — OPV Dose 2 overdue by 7 days

  Dropout Analysis (Rampur village):
  ├── Pentavalent: 92% started → 78% completed → 15.2% dropout
  ├── OPV: 95% started → 81% completed → 14.7% dropout
  └── Measles: 85% started → 72% completed → 15.3% dropout
```
