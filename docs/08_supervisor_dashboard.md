# Use Case: Supervisor Dashboard

## Problem Statement

ASHA supervisors (ANMs, Block Health Officers) need a consolidated view of maternal and child health indicators across their jurisdiction to identify high-risk patients, monitor immunization coverage, track scheme enrollment, and evaluate ASHA worker performance. Currently, this data is scattered across paper registers, making it impossible to get real-time situational awareness or prioritize field interventions.

---

## Solution

Asha AI provides a **real-time supervisor analytics dashboard** built with Gradio and Plotly, powered by Databricks SQL queries over Delta Lake. It surfaces key performance indicators (KPIs), risk distribution charts, village-level heatmaps, immunization coverage, and high-risk patient tables — giving supervisors an actionable overview without manual data aggregation.

---

## Dashboard Components

### 1. KPI Cards

| Metric | Source Query | Purpose |
|--------|-------------|---------|
| Total Patients | `COUNT(*) FROM patients` | Population coverage |
| RED Alerts (Active) | `COUNT(*) FROM triage_alerts WHERE risk_level = 'RED'` | Emergency cases requiring action |
| Overdue Vaccines | `COUNT(*) FROM immunizations WHERE date_due < today AND date_given IS NULL` | Immunization gaps |
| Scheme Enrollments | `COUNT(*) FROM scheme_applications WHERE eligible = true` | Welfare coverage |

### 2. Risk Distribution (Pie Chart)

- Segments: RED, YELLOW, GREEN
- Source: `triage_alerts GROUP BY risk_level`
- Color-coded: Red (#dc2626), Yellow (#f59e0b), Green (#22c55e)

### 3. Village-Level Risk Heatmap (Stacked Bar Chart)

- X-axis: Villages
- Y-axis: Patient count
- Stacks: RED, YELLOW, GREEN per village
- Source: `triage_alerts JOIN patients GROUP BY village, risk_level`

### 4. Immunization Coverage (Grouped Bar Chart)

- X-axis: Vaccine types (BCG, OPV, Pentavalent, Measles, etc.)
- Y-axis: Coverage percentage
- Bars: Given vs Due
- Source: `immunizations GROUP BY vaccine_name`

### 5. High-Risk Patient Table

| Column | Description |
|--------|-------------|
| Patient Name | From `patients` table |
| Age | Patient age |
| Village | Location for field visit planning |
| Risk Level | RED or YELLOW |
| Danger Sign | Primary clinical concern |
| Action Required | Recommended intervention |
| Assigned Doctor | Name + phone for referral |
| Urgency | Hours until intervention needed |

### 6. ASHA Activity Summary

- Visits logged per ASHA per week
- Patients registered per ASHA
- Triage alerts generated per ASHA

---

## Data Flow

```
Delta Lake Tables
├── patients
├── visits
├── immunizations
├── triage_alerts
├── doctors
├── phc_facilities
└── scheme_applications
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Dashboard Helper Queries                         │
│  (src/dashboard_helpers.py)                       │
│                                                   │
│  ├── get_high_risk_patients(spark)                │
│  ├── get_risk_level_distribution(spark)           │
│  ├── get_village_risk_heatmap(spark)              │
│  ├── get_immunization_coverage(spark)             │
│  ├── get_scheme_enrollment_summary(spark)         │
│  └── get_asha_activity_summary(spark)             │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Pandas DataFrames                                │
│  (Spark → toPandas() for Plotly/Gradio)           │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Plotly Visualization                             │
│  ├── px.pie() — risk distribution                 │
│  ├── px.bar() — village heatmap (stacked)         │
│  ├── px.bar() — immunization coverage (grouped)   │
│  └── go.Figure() — custom KPI cards               │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Gradio Blocks UI                                 │
│  ├── gr.Row() — KPI metric cards                  │
│  ├── gr.Plot() — interactive Plotly charts         │
│  └── gr.Dataframe() — high-risk patient table     │
└──────────────────────────────────────────────────┘
```

---

## Architecture

| Layer | Component | Technology |
|-------|-----------|-----------|
| **Data Store** | Patient, visit, triage, immunization records | Delta Lake (Unity Catalog) |
| **Query Engine** | Analytical aggregations | Databricks SQL (Spark SQL) |
| **Data Transform** | Spark DataFrames → Pandas | PySpark + Pandas |
| **Visualization** | Interactive charts | Plotly Express + Graph Objects |
| **UI Framework** | Dashboard layout | Gradio Blocks |
| **Compute** | Query execution | Databricks Serverless Compute |

---

## Key Source Files

| File | Purpose |
|------|---------|
| `notebooks/09_supervisor_dashboard.py` | Full dashboard UI with Plotly charts and Gradio layout |
| `src/dashboard_helpers.py` | SQL query functions for all dashboard metrics |
| `config/settings.py` | Database name and schema configuration |

---

## Demo

```
Launch: Run notebook 09_supervisor_dashboard.py

Dashboard shows:
  KPI Cards:
  ├── Total Patients: 500
  ├── RED Alerts: 23
  ├── Overdue Vaccines: 87
  └── Scheme Enrollments: 312

  Risk Pie: RED 4.6% | YELLOW 18.2% | GREEN 77.2%

  Village Heatmap: Rampur (highest RED), Sultanpur, Jaunpur, ...

  High-Risk Table:
  ├── Sunita Devi | 24 | Rampur | RED | Pre-eclampsia | Immediate referral | Dr. Meena Tripathi
  └── Kavita Kumari | 19 | Sultanpur | RED | Severe anemia | Refer within 2 hours | Dr. Anil Sharma
```
