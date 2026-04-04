# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Scheme Eligibility Demo
# MAGIC ML-assisted evaluation of PMMVY, JSY, and JSSK welfare schemes with rule-based fallback.

# COMMAND ----------

# MAGIC %pip install requests pyyaml
# MAGIC %restart_python

# COMMAND ----------

import sys, os
repo_root = os.path.dirname(os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
))
sys.path.insert(0, f"/Workspace{repo_root}")

from src.models.indic_llm import IndicLLM
from src.scheme_eligibility import SchemeEngine

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Initialize Scheme Engine

# COMMAND ----------

llm = IndicLLM()
engine = SchemeEngine(llm=llm)
print(f"LLM API available: {llm.is_loaded}")
print(f"Method: {'ML-primary (Sarvam)' if llm.is_loaded else 'Rule-based fallback'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Evaluate All Patients

# COMMAND ----------

patients = spark.table("workspace.asha_copilot.patients").limit(50).collect()

all_results = []
for p in patients:
    patient_dict = p.asDict()
    results = engine.evaluate(patient_dict)
    for r in results:
        all_results.append({
            "patient_id": patient_dict["patient_id"],
            "patient_name": patient_dict["name"],
            "village": patient_dict["village"],
            "scheme_id": r.scheme_id,
            "eligible": r.eligible,
            "missing": ", ".join(r.missing_requirements) if r.missing_requirements else "None",
            "amount": r.amount,
            "notes": r.notes,
        })

results_df = spark.createDataFrame(all_results)
print(f"Evaluated {len(patients)} patients across 3 schemes → {len(all_results)} results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Eligibility Summary

# COMMAND ----------

display(
    results_df.groupBy("scheme_id", "eligible").count().orderBy("scheme_id", "eligible")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Specific Cases

# COMMAND ----------

# MAGIC %md
# MAGIC ### Eligible for PMMVY (first pregnancy, has all documents)

# COMMAND ----------

display(
    results_df.filter("scheme_id = 'PMMVY' AND eligible = true").limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Eligible for JSY — amounts by state

# COMMAND ----------

display(
    results_df.filter("scheme_id = 'JSY' AND eligible = true")
    .select("patient_name", "village", "amount", "notes")
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Patients with Missing Documents

# COMMAND ----------

display(
    results_df.filter("eligible = false AND missing != 'None'")
    .select("patient_name", "scheme_id", "missing")
    .limit(15)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Actionable Summary for ASHA

# COMMAND ----------

from pyspark.sql.functions import col, explode, split

missing_breakdown = (
    results_df
    .filter("eligible = false AND missing != 'None'")
    .select("scheme_id", explode(split(col("missing"), ", ")).alias("missing_doc"))
    .groupBy("scheme_id", "missing_doc")
    .count()
    .orderBy("scheme_id", "count", ascending=False)
)
display(missing_breakdown)

# COMMAND ----------

print("=" * 60)
print("ACTIONABLE SUMMARY FOR ASHA SUPERVISOR")
print("=" * 60)
for scheme in ["PMMVY", "JSY", "JSSK"]:
    eligible = results_df.filter(f"scheme_id = '{scheme}' AND eligible = true").count()
    ineligible = results_df.filter(f"scheme_id = '{scheme}' AND eligible = false").count()
    print(f"\n{scheme}:")
    print(f"  Eligible: {eligible} patients")
    print(f"  Need action: {ineligible} patients")
