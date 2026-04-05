# Databricks notebook source

# MAGIC %md
# MAGIC # 09 — Supervisor Dashboard
# MAGIC Real-time analytics for ASHA supervisors: risk overview, immunization gaps, scheme enrollment, ASHA activity.

# COMMAND ----------

# MAGIC %pip install gradio plotly
# MAGIC %restart_python

# COMMAND ----------

import sys, os
repo_root = os.path.dirname(os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
))
sys.path.insert(0, f"/Workspace{repo_root}")

from src.dashboard_helpers import (
    get_high_risk_patients,
    get_risk_level_distribution,
    get_immunization_coverage,
    get_immunization_overdue,
    get_scheme_summary,
    get_village_risk_summary,
    get_asha_activity_summary,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Dashboard Data

# COMMAND ----------

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Load all data
risk_dist = get_risk_level_distribution(spark).toPandas()
high_risk = get_high_risk_patients(spark).toPandas()
imm_coverage = get_immunization_coverage(spark).toPandas()
imm_overdue = get_immunization_overdue(spark).toPandas()
scheme_summary = get_scheme_summary(spark).toPandas()
village_risk = get_village_risk_summary(spark).toPandas()
activity = get_asha_activity_summary(spark).toPandas()

print("Dashboard data loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Dashboard

# COMMAND ----------

import gradio as gr

def make_risk_pie():
    if risk_dist.empty:
        return None
    colors = {"RED": "#FF4444", "YELLOW": "#FFAA00", "GREEN": "#44BB44"}
    fig = px.pie(
        risk_dist, values="count", names="risk_level",
        title="Risk Level Distribution",
        color="risk_level",
        color_discrete_map=colors,
    )
    fig.update_layout(height=350)
    return fig

def make_imm_bar():
    if imm_coverage.empty:
        return None
    fig = px.bar(
        imm_coverage, x="vaccine_code", y="coverage_pct",
        title="Immunization Coverage by Vaccine (%)",
        color="coverage_pct",
        color_continuous_scale=["#FF4444", "#FFAA00", "#44BB44"],
        range_color=[0, 100],
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)
    return fig

def make_village_bar():
    if village_risk.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(name="RED", x=village_risk["village"], y=village_risk["red_count"],
                         marker_color="#FF4444"))
    fig.add_trace(go.Bar(name="YELLOW", x=village_risk["village"], y=village_risk["yellow_count"],
                         marker_color="#FFAA00"))
    fig.add_trace(go.Bar(name="GREEN", x=village_risk["village"], y=village_risk["green_count"],
                         marker_color="#44BB44"))
    fig.update_layout(barmode="stack", title="Village-Level Risk Distribution", height=400)
    return fig

def make_scheme_bar():
    if scheme_summary.empty:
        return None
    fig = px.bar(
        scheme_summary, x="scheme_id", y="count", color="eligibility_status",
        title="Scheme Applications by Status",
        barmode="group",
        color_discrete_map={"Eligible": "#44BB44", "Pending Documents": "#FFAA00", "Ineligible": "#FF4444"},
    )
    fig.update_layout(height=350)
    return fig


with gr.Blocks(title="Asha AI — Supervisor Dashboard", theme=gr.themes.Soft()) as dashboard:
    gr.Markdown("# Asha AI — Supervisor Dashboard")
    gr.Markdown("Real-time overview of maternal & child health indicators across your jurisdiction")

    # ---- KPI Cards ----
    with gr.Row():
        for _, row in activity.iterrows():
            with gr.Column(scale=1):
                gr.Markdown(f"### {row['metric']}\n# {row['value']}")

    # ---- Charts Row 1 ----
    with gr.Row():
        with gr.Column(scale=1):
            gr.Plot(value=make_risk_pie, label="Risk Distribution")
        with gr.Column(scale=2):
            gr.Plot(value=make_village_bar, label="Village Risk Heatmap")

    # ---- Charts Row 2 ----
    with gr.Row():
        with gr.Column(scale=1):
            gr.Plot(value=make_imm_bar, label="Immunization Coverage")
        with gr.Column(scale=1):
            gr.Plot(value=make_scheme_bar, label="Scheme Status")

    # ---- High Risk Patients Table ----
    gr.Markdown("## High Risk Patients — Needs Immediate Attention")
    if not high_risk.empty:
        gr.Dataframe(
            value=high_risk[["patient_name", "age", "village", "risk_level",
                             "sign_name", "action", "assigned_phc_name",
                             "assigned_doctor_name", "assigned_doctor_phone"]].head(20),
            label="RED & YELLOW Patients",
        )
    else:
        gr.Markdown("No high-risk patients found.")

    # ---- Overdue Immunizations ----
    gr.Markdown("## Overdue Immunizations — Follow-Up Required")
    if not imm_overdue.empty:
        gr.Dataframe(
            value=imm_overdue[["child_name", "vaccine_code", "due_date", "village"]].head(20),
            label="Overdue Vaccines",
        )
    else:
        gr.Markdown("No overdue immunizations.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Launch Dashboard

# COMMAND ----------

dashboard.launch()
