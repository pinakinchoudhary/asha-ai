# Databricks notebook source

# MAGIC %md
# MAGIC # 01 — Generate Synthetic Data
# MAGIC Populates all Delta tables with realistic Indian maternal health records.
# MAGIC Includes PHC facilities, doctors, patients, visits, immunizations, and scheme applications.

# COMMAND ----------

# MAGIC %pip install faker pyyaml
# MAGIC %restart_python

# COMMAND ----------

import sys, os
# Add repo root to path for imports
repo_root = os.path.dirname(os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
))
sys.path.insert(0, f"/Workspace{repo_root}")

# COMMAND ----------

import yaml
from pyspark.sql import Row

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load PHC Facilities & Doctors from YAML config

# COMMAND ----------

config_path = f"/Workspace{repo_root}/config/phc_doctors.yaml"
with open(config_path, "r") as f:
    phc_config = yaml.safe_load(f)

# Insert PHC facilities
phc_rows = []
for fac in phc_config["phc_facilities"]:
    phc_rows.append(Row(
        phc_id=fac["phc_id"],
        name=fac["name"],
        district=fac["district"],
        state=fac["state"],
        type=fac["type"],
        latitude=float(fac["latitude"]),
        longitude=float(fac["longitude"]),
        has_blood_bank=fac["has_blood_bank"],
        has_surgery=fac["has_surgery"],
        beds=int(fac["beds"]),  # explicit int to avoid Long vs Integer schema mismatch
        villages_served=fac["villages_served"],
    ))
from pyspark.sql.functions import col
phc_df = spark.createDataFrame(phc_rows).withColumn("beds", col("beds").cast("int"))
phc_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("workspace.asha_copilot.phc_facilities")
print(f"PHC Facilities: {phc_df.count()} rows written")
display(phc_df)

# COMMAND ----------

# Insert Doctors
doc_rows = []
for doc in phc_config["doctors"]:
    doc_rows.append(Row(
        doctor_id=doc["doctor_id"],
        name=doc["name"],
        phc_id=doc["phc_id"],
        specialization=doc["specialization"],
        phone=doc["phone"],
        available_days=doc["available_days"],
    ))
doc_df = spark.createDataFrame(doc_rows)
doc_df.write.format("delta").mode("overwrite").saveAsTable("workspace.asha_copilot.doctors")
print(f"Doctors: {doc_df.count()} rows written")
display(doc_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate Synthetic Patients

# COMMAND ----------

from src.synthetic_data import generate_patients, generate_visits, generate_immunizations, generate_scheme_applications

patients_df = generate_patients(spark, n=500)
patients_df.write.format("delta").mode("overwrite").saveAsTable("workspace.asha_copilot.patients")
print(f"Patients: {patients_df.count()} rows written")
display(patients_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate ANC Visits

# COMMAND ----------

visits_df = generate_visits(spark, patients_df)
visits_df.write.format("delta").mode("overwrite").saveAsTable("workspace.asha_copilot.visits")
print(f"Visits: {visits_df.count()} rows written")

# Show danger sign distribution
print("\nDanger sign visits:")
display(visits_df.filter("referral_flag = true").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Generate Immunization Records

# COMMAND ----------

imm_df = generate_immunizations(spark, patients_df)
if imm_df:
    imm_df.write.format("delta").mode("overwrite").saveAsTable("workspace.asha_copilot.immunizations")
    print(f"Immunizations: {imm_df.count()} rows written")
    print(f"Overdue vaccines: {imm_df.filter('missed_flag = true').count()}")
    display(imm_df.filter("missed_flag = true").limit(10))
else:
    print("No immunization records generated (no babies born yet in synthetic data)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate Scheme Applications

# COMMAND ----------

scheme_df = generate_scheme_applications(spark, patients_df)
scheme_df.write.format("delta").mode("overwrite").saveAsTable("workspace.asha_copilot.scheme_applications")
print(f"Scheme Applications: {scheme_df.count()} rows written")
display(
    scheme_df.groupBy("scheme_id", "eligibility_status").count().orderBy("scheme_id")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("SYNTHETIC DATA GENERATION COMPLETE")
print("=" * 60)
for table in ["phc_facilities", "doctors", "patients", "visits", "immunizations", "scheme_applications"]:
    count = spark.table(f"workspace.asha_copilot.{table}").count()
    print(f"  {table:30s} → {count:,} rows")
