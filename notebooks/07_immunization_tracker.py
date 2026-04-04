# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Immunization Tracker
# MAGIC Tracks Universal Immunization Programme (UIP) schedules and flags overdue vaccines.

# COMMAND ----------

import sys, os
repo_root = os.path.dirname(os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
))
sys.path.insert(0, f"/Workspace{repo_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Immunization Overview

# COMMAND ----------

imm_df = spark.table("workspace.asha_copilot.immunizations")
total = imm_df.count()
administered = imm_df.filter("administered_date IS NOT NULL").count()
missed = imm_df.filter("missed_flag = true").count()

print(f"Total vaccine records: {total}")
print(f"Administered: {administered} ({administered*100//total if total else 0}%)")
print(f"Missed/Overdue: {missed} ({missed*100//total if total else 0}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Coverage by Vaccine

# COMMAND ----------

display(
    spark.sql("""
        SELECT
            vaccine_code,
            COUNT(*) as total,
            SUM(CASE WHEN administered_date IS NOT NULL THEN 1 ELSE 0 END) as given,
            SUM(CASE WHEN missed_flag = true THEN 1 ELSE 0 END) as overdue,
            ROUND(
                SUM(CASE WHEN administered_date IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
                1
            ) as coverage_pct
        FROM workspace.asha_copilot.immunizations
        GROUP BY vaccine_code
        ORDER BY
            CASE
                WHEN vaccine_code LIKE 'BCG%' THEN 1
                WHEN vaccine_code LIKE 'OPV-0%' THEN 2
                WHEN vaccine_code LIKE 'Hep%' THEN 3
                WHEN vaccine_code LIKE 'OPV-1%' THEN 4
                WHEN vaccine_code LIKE 'Penta%1' THEN 5
                WHEN vaccine_code LIKE 'OPV-2%' THEN 6
                WHEN vaccine_code LIKE 'Penta%2' THEN 7
                WHEN vaccine_code LIKE 'OPV-3%' THEN 8
                WHEN vaccine_code LIKE 'Penta%3' THEN 9
                WHEN vaccine_code LIKE 'MR-1%' THEN 10
                ELSE 20
            END
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Overdue Children — Needs Immediate Follow-Up

# COMMAND ----------

display(
    spark.sql("""
        SELECT
            i.child_name,
            i.vaccine_code,
            i.due_date,
            DATEDIFF(current_date(), i.due_date) as days_overdue,
            i.village,
            p.name as mother_name,
            p.phone as mother_phone,
            f.name as nearest_facility
        FROM workspace.asha_copilot.immunizations i
        JOIN workspace.asha_copilot.patients p ON i.mother_id = p.patient_id
        LEFT JOIN workspace.asha_copilot.phc_facilities f ON i.nearest_phc_id = f.phc_id
        WHERE i.missed_flag = true
        ORDER BY DATEDIFF(current_date(), i.due_date) DESC
        LIMIT 20
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Overdue by Village (for ASHA routing)

# COMMAND ----------

display(
    spark.sql("""
        SELECT
            village,
            COUNT(DISTINCT child_id) as children_affected,
            COUNT(*) as total_overdue_doses,
            COLLECT_SET(vaccine_code) as overdue_vaccines
        FROM workspace.asha_copilot.immunizations
        WHERE missed_flag = true
        GROUP BY village
        ORDER BY children_affected DESC
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Dropout Analysis (BCG → MR-1)

# COMMAND ----------

display(
    spark.sql("""
        WITH bcg AS (
            SELECT child_id
            FROM workspace.asha_copilot.immunizations
            WHERE vaccine_code = 'BCG' AND administered_date IS NOT NULL
        ),
        mr1 AS (
            SELECT child_id
            FROM workspace.asha_copilot.immunizations
            WHERE vaccine_code = 'MR-1' AND administered_date IS NOT NULL
        )
        SELECT
            COUNT(DISTINCT b.child_id) as received_bcg,
            COUNT(DISTINCT m.child_id) as received_mr1,
            COUNT(DISTINCT b.child_id) - COUNT(DISTINCT m.child_id) as dropout_count,
            ROUND(
                (COUNT(DISTINCT b.child_id) - COUNT(DISTINCT m.child_id)) * 100.0
                / NULLIF(COUNT(DISTINCT b.child_id), 0),
                1
            ) as dropout_rate_pct
        FROM bcg b
        LEFT JOIN mr1 m ON b.child_id = m.child_id
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("IMMUNIZATION TRACKER SUMMARY")
print("=" * 60)
print(f"Total records: {total}")
print(f"Coverage rate: {administered*100//total if total else 0}%")
print(f"Overdue doses: {missed}")
villages = spark.sql("""
    SELECT COUNT(DISTINCT village) as v
    FROM workspace.asha_copilot.immunizations WHERE missed_flag = true
""").collect()[0]["v"]
print(f"Villages with overdue: {villages}")
