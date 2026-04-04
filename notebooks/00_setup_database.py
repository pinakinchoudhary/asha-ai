# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Setup Database & Tables
# MAGIC Creates the `asha_copilot` schema and all Delta tables in `workspace` catalog.
# MAGIC **Idempotent** — safe to run multiple times.

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.asha_copilot")
spark.sql("USE workspace.asha_copilot")
print("Schema workspace.asha_copilot ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Delta Tables

# COMMAND ----------

# Patients table
spark.sql("""
CREATE TABLE IF NOT EXISTS patients (
    patient_id STRING,
    name STRING,
    age INT,
    husband_name STRING,
    state STRING,
    district STRING,
    village STRING,
    nearest_phc_id STRING,
    caste_category STRING,
    bpl_status BOOLEAN,
    blood_group STRING,
    gravida INT,
    para INT,
    abortions INT,
    lmp STRING,
    edd STRING,
    risk_flags ARRAY<STRING>,
    aadhaar_registered BOOLEAN,
    bank_account BOOLEAN,
    phone STRING,
    registered_date STRING
)
USING DELTA
""")
print("Table: patients ✓")

# COMMAND ----------

# Visits table
spark.sql("""
CREATE TABLE IF NOT EXISTS visits (
    visit_id STRING,
    patient_id STRING,
    visit_number INT,
    visit_date STRING,
    gestational_age_weeks INT,
    weight_kg FLOAT,
    fundal_height_cm FLOAT,
    bp_systolic INT,
    bp_diastolic INT,
    hemoglobin_g_dl FLOAT,
    temperature_c FLOAT,
    danger_signs_observed ARRAY<STRING>,
    clinical_notes STRING,
    referral_flag BOOLEAN
)
USING DELTA
""")
print("Table: visits ✓")

# COMMAND ----------

# Immunizations table
spark.sql("""
CREATE TABLE IF NOT EXISTS immunizations (
    record_id STRING,
    child_id STRING,
    mother_id STRING,
    child_name STRING,
    date_of_birth STRING,
    vaccine_code STRING,
    due_date STRING,
    administered_date STRING,
    missed_flag BOOLEAN,
    village STRING,
    nearest_phc_id STRING
)
USING DELTA
""")
print("Table: immunizations ✓")

# COMMAND ----------

# Scheme applications table
spark.sql("""
CREATE TABLE IF NOT EXISTS scheme_applications (
    application_id STRING,
    patient_id STRING,
    scheme_id STRING,
    eligibility_status STRING,
    missing_documents ARRAY<STRING>,
    disbursement_amount INT,
    applied_date STRING
)
USING DELTA
""")
print("Table: scheme_applications ✓")

# COMMAND ----------

# Triage alerts table
spark.sql("""
CREATE TABLE IF NOT EXISTS triage_alerts (
    alert_id STRING,
    patient_id STRING,
    visit_id STRING,
    risk_level STRING,
    sign_name STRING,
    severity STRING,
    action STRING,
    urgency_hours INT,
    confidence FLOAT,
    method STRING,
    assigned_phc_id STRING,
    assigned_phc_name STRING,
    assigned_doctor_id STRING,
    assigned_doctor_name STRING,
    assigned_doctor_phone STRING,
    alert_date STRING
)
USING DELTA
""")
print("Table: triage_alerts ✓")

# COMMAND ----------

# Doctors table
spark.sql("""
CREATE TABLE IF NOT EXISTS doctors (
    doctor_id STRING,
    name STRING,
    phc_id STRING,
    specialization STRING,
    phone STRING,
    available_days ARRAY<STRING>
)
USING DELTA
""")
print("Table: doctors ✓")

# COMMAND ----------

# PHC Facilities table
spark.sql("""
CREATE TABLE IF NOT EXISTS phc_facilities (
    phc_id STRING,
    name STRING,
    district STRING,
    state STRING,
    type STRING,
    latitude DOUBLE,
    longitude DOUBLE,
    has_blood_bank BOOLEAN,
    has_surgery BOOLEAN,
    beds INT,
    villages_served ARRAY<STRING>
)
USING DELTA
""")
print("Table: phc_facilities ✓")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify All Tables

# COMMAND ----------

tables = spark.sql("SHOW TABLES IN workspace.asha_copilot").collect()
print(f"\n{'='*50}")
print(f"ASHA Copilot Schema: {len(tables)} tables created")
print(f"{'='*50}")
for t in tables:
    print(f"  - {t['tableName']}")
