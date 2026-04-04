"""
Dashboard query helpers for the Supervisor Dashboard.

All functions accept a SparkSession and return DataFrames or dicts
suitable for Gradio/Plotly visualization.
"""

import logging

logger = logging.getLogger(__name__)

DB = "workspace.asha_copilot"


def get_high_risk_patients(spark):
    """Get patients with RED/YELLOW triage alerts and assigned doctors."""
    return spark.sql(f"""
        SELECT
            p.name AS patient_name,
            p.age,
            p.village,
            p.district,
            t.risk_level,
            t.sign_name,
            t.action,
            t.urgency_hours,
            t.assigned_phc_name,
            t.assigned_doctor_name,
            t.assigned_doctor_phone,
            t.alert_date
        FROM {DB}.triage_alerts t
        JOIN {DB}.patients p ON t.patient_id = p.patient_id
        WHERE t.risk_level IN ('RED', 'YELLOW')
        ORDER BY
            CASE t.risk_level WHEN 'RED' THEN 0 WHEN 'YELLOW' THEN 1 ELSE 2 END,
            t.alert_date DESC
    """)


def get_risk_level_distribution(spark):
    """Get count of patients by risk level."""
    return spark.sql(f"""
        SELECT risk_level, COUNT(*) as count
        FROM {DB}.triage_alerts
        GROUP BY risk_level
        ORDER BY
            CASE risk_level WHEN 'RED' THEN 0 WHEN 'YELLOW' THEN 1 ELSE 2 END
    """)


def get_immunization_coverage(spark):
    """Get immunization completion rates by vaccine."""
    return spark.sql(f"""
        SELECT
            vaccine_code,
            COUNT(*) as total_records,
            SUM(CASE WHEN administered_date IS NOT NULL THEN 1 ELSE 0 END) as administered,
            SUM(CASE WHEN missed_flag = true THEN 1 ELSE 0 END) as missed,
            ROUND(
                SUM(CASE WHEN administered_date IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
                1
            ) as coverage_pct
        FROM {DB}.immunizations
        GROUP BY vaccine_code
        ORDER BY vaccine_code
    """)


def get_immunization_overdue(spark):
    """Get children with overdue vaccinations."""
    return spark.sql(f"""
        SELECT
            child_name,
            vaccine_code,
            due_date,
            village,
            nearest_phc_id
        FROM {DB}.immunizations
        WHERE missed_flag = true
        ORDER BY due_date
    """)


def get_scheme_summary(spark):
    """Get scheme eligibility summary."""
    return spark.sql(f"""
        SELECT
            scheme_id,
            eligibility_status,
            COUNT(*) as count,
            SUM(disbursement_amount) as total_amount
        FROM {DB}.scheme_applications
        GROUP BY scheme_id, eligibility_status
        ORDER BY scheme_id, eligibility_status
    """)


def get_village_risk_summary(spark):
    """Get village-level risk heatmap data."""
    return spark.sql(f"""
        SELECT
            p.village,
            p.district,
            p.state,
            COUNT(DISTINCT p.patient_id) as total_patients,
            SUM(CASE WHEN t.risk_level = 'RED' THEN 1 ELSE 0 END) as red_count,
            SUM(CASE WHEN t.risk_level = 'YELLOW' THEN 1 ELSE 0 END) as yellow_count,
            SUM(CASE WHEN t.risk_level = 'GREEN' THEN 1 ELSE 0 END) as green_count
        FROM {DB}.patients p
        LEFT JOIN {DB}.triage_alerts t ON p.patient_id = t.patient_id
        GROUP BY p.village, p.district, p.state
        ORDER BY red_count DESC, yellow_count DESC
    """)


def get_asha_activity_summary(spark):
    """Get ASHA activity metrics (visits logged, patients registered)."""
    return spark.sql(f"""
        SELECT
            'Total Patients' as metric, CAST(COUNT(*) AS STRING) as value
        FROM {DB}.patients
        UNION ALL
        SELECT
            'Total Visits', CAST(COUNT(*) AS STRING)
        FROM {DB}.visits
        UNION ALL
        SELECT
            'Red Alerts', CAST(COUNT(*) AS STRING)
        FROM {DB}.triage_alerts WHERE risk_level = 'RED'
        UNION ALL
        SELECT
            'Overdue Vaccines', CAST(COUNT(*) AS STRING)
        FROM {DB}.immunizations WHERE missed_flag = true
        UNION ALL
        SELECT
            'Scheme-Eligible', CAST(COUNT(*) AS STRING)
        FROM {DB}.scheme_applications WHERE eligibility_status = 'Eligible'
    """)


def get_patient_details(spark, patient_id: str):
    """Get full patient details with visit history."""
    patient = spark.sql(f"""
        SELECT * FROM {DB}.patients WHERE patient_id = '{patient_id}'
    """).collect()

    visits = spark.sql(f"""
        SELECT * FROM {DB}.visits
        WHERE patient_id = '{patient_id}'
        ORDER BY visit_date DESC
    """).collect()

    triage = spark.sql(f"""
        SELECT * FROM {DB}.triage_alerts
        WHERE patient_id = '{patient_id}'
        ORDER BY alert_date DESC
    """).collect()

    return {
        "patient": patient[0].asDict() if patient else {},
        "visits": [v.asDict() for v in visits],
        "triage_alerts": [t.asDict() for t in triage],
    }
