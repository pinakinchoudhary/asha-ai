# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Clinical Triage Demo
# MAGIC ML-primary triage with rule-based fallback and automatic doctor/PHC assignment.
# MAGIC
# MAGIC **Flow**: Patient vitals → Sarvam AI Assessment → Risk Classification (RED/YELLOW/GREEN) → Doctor + PHC Assignment

# COMMAND ----------

# MAGIC %pip install pyyaml requests
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Load API Keys
import os

# Load API keys from secrets (using same keys as notebook 02)
try:
    os.environ["SARVAM_API_KEY"] = dbutils.secrets.get(scope="asha-ai", key="sarvam-api-key")
    print("✓ SARVAM_API_KEY loaded")
except: pass

try:
    os.environ["HF_TOKEN"] = dbutils.secrets.get(scope="asha-ai", key="hf-token")
    print("✓ HF_TOKEN loaded")
except: pass

try:
    os.environ["GROQ_API_KEY"] = dbutils.secrets.get(scope="asha-ai", key="groq-api-key")
    print("✓ GROQ_API_KEY loaded")
except: pass

# COMMAND ----------

import sys, os, uuid
from datetime import date

repo_root = os.path.dirname(os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
))
sys.path.insert(0, f"/Workspace{repo_root}")

from src.models.indic_llm import IndicLLM
from src.triage_engine import TriageEngine

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Initialize Triage Engine

# COMMAND ----------

llm = IndicLLM()
rules_path = f"/Workspace{repo_root}/config/danger_signs.yaml"
engine = TriageEngine(llm=llm, rules_path=rules_path)

print(f"LLM API available: {llm.is_loaded}")
print(f"Method: {'ML-primary (Sarvam)' if llm.is_loaded else 'Rule-based fallback'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Run Triage on Synthetic Visit Data

# COMMAND ----------

visits_df = spark.table("workspace.asha_copilot.visits")
patients_df = spark.table("workspace.asha_copilot.patients")

# Join to get village/PHC context
visits_with_context = visits_df.join(
    patients_df.select("patient_id", "name", "village", "nearest_phc_id"),
    on="patient_id"
)

print(f"Total visits to triage: {visits_with_context.count()}")

# COMMAND ----------

# Process a sample of visits (full dataset can be done in production)
sample_visits = visits_with_context.limit(100).collect()

triage_results = []
for visit in sample_visits:
    visit_dict = visit.asDict()
    result = engine.triage(visit_dict, spark=spark)

    alert_row = {
        "alert_id": str(uuid.uuid4()),
        "patient_id": visit_dict["patient_id"],
        "visit_id": visit_dict["visit_id"],
        "risk_level": result.risk_level,
        "sign_name": result.alerts[0].sign_name if result.alerts else "none",
        "severity": result.alerts[0].severity if result.alerts else "low",
        "action": result.alerts[0].action if result.alerts else "Continue monitoring",
        "urgency_hours": result.alerts[0].urgency_hours if result.alerts else 168,
        "confidence": result.alerts[0].confidence if result.alerts else 1.0,
        "method": result.method,
        "assigned_phc_id": result.assignment.phc_id if result.assignment else "",
        "assigned_phc_name": result.assignment.phc_name if result.assignment else "",
        "assigned_doctor_id": result.assignment.doctor_id if result.assignment else "",
        "assigned_doctor_name": result.assignment.doctor_name if result.assignment else "",
        "assigned_doctor_phone": result.assignment.doctor_phone if result.assignment else "",
        "alert_date": str(date.today()),
    }
    triage_results.append(alert_row)

print(f"Triaged {len(triage_results)} visits")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write Triage Alerts to Delta

# COMMAND ----------

from pyspark.sql.functions import col

alerts_df = spark.createDataFrame(triage_results)
# Cast columns to match existing table schema
alerts_df = alerts_df.withColumn("urgency_hours", col("urgency_hours").cast("int")) \
                     .withColumn("confidence", col("confidence").cast("float"))
alerts_df.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.asha_copilot.triage_alerts"
)
print("Triage alerts written to Delta table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Risk Level Distribution

# COMMAND ----------

display(
    spark.sql("""
        SELECT risk_level, method, COUNT(*) as count
        FROM workspace.asha_copilot.triage_alerts
        GROUP BY risk_level, method
        ORDER BY
            CASE risk_level WHEN 'RED' THEN 0 WHEN 'YELLOW' THEN 1 ELSE 2 END
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. RED Cases — Doctor & PHC Assignments

# COMMAND ----------

display(
    spark.sql("""
        SELECT
            p.name AS patient,
            p.village,
            t.risk_level,
            t.sign_name,
            t.action,
            t.assigned_phc_name AS facility,
            t.assigned_doctor_name AS doctor,
            t.assigned_doctor_phone AS phone,
            t.method
        FROM workspace.asha_copilot.triage_alerts t
        JOIN workspace.asha_copilot.patients p ON t.patient_id = p.patient_id
        WHERE t.risk_level = 'RED'
        ORDER BY t.urgency_hours ASC
        LIMIT 20
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Example: Single Patient Journey

# COMMAND ----------

patient_with_visits = spark.sql("""
    SELECT patient_id, COUNT(*) as n_visits
    FROM workspace.asha_copilot.visits
    GROUP BY patient_id
    HAVING COUNT(*) >= 3
    LIMIT 1
""").collect()

if patient_with_visits:
    pid = patient_with_visits[0]["patient_id"]
    print(f"Patient journey for ID: {pid[:8]}...")
    display(
        spark.sql(f"""
            SELECT
                v.visit_number,
                v.visit_date,
                v.bp_systolic, v.bp_diastolic,
                v.hemoglobin_g_dl,
                v.clinical_notes,
                t.risk_level,
                t.sign_name,
                t.assigned_doctor_name
            FROM workspace.asha_copilot.visits v
            LEFT JOIN workspace.asha_copilot.triage_alerts t
                ON v.visit_id = t.visit_id
            WHERE v.patient_id = '{pid}'
            ORDER BY v.visit_number
        """)
    )
