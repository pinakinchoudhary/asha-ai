"""
Clinical Triage Engine.

ML-PRIMARY with rule-based fallback. Assigns doctor + PHC based on severity.
Hard-coded red flags ALWAYS override any classification.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TriageAlert:
    sign_name: str
    severity: str  # "critical", "high", "moderate", "low"
    action: str
    urgency_hours: int
    confidence: float = 1.0


@dataclass
class AssignmentResult:
    phc_id: str = ""
    phc_name: str = ""
    phc_type: str = ""
    doctor_id: str = ""
    doctor_name: str = ""
    doctor_phone: str = ""
    doctor_specialization: str = ""
    distance_km: Optional[float] = None


@dataclass
class TriageResult:
    risk_level: str  # "GREEN", "YELLOW", "RED"
    alerts: list = field(default_factory=list)
    assignment: Optional[AssignmentResult] = None
    method: str = "ml"  # "ml" or "rule_fallback"

    def to_dict(self):
        d = {
            "risk_level": self.risk_level,
            "alerts": [asdict(a) for a in self.alerts],
            "method": self.method,
        }
        if self.assignment:
            d["assignment"] = asdict(self.assignment)
        return d


# ---------------------------------------------------------------------------
# Hard-coded red flags — these ALWAYS trigger RED, bypass everything
# ---------------------------------------------------------------------------
HARD_RED_FLAGS = {
    "convulsions": ["convulsion", "seizure", "fits", "daurey", "jhatke", "mirgi"],
    "severe_bleeding": ["severe bleeding", "hemorrhage", "bahut khoon", "bahut bleeding"],
    "chest_indrawing": ["chest indrawing", "chhati dhans"],
    "foul_lochia": ["foul smelling lochia", "foul discharge", "badbu wala discharge"],
    "yellow_palms": ["yellow palms", "yellow soles", "peeli hatheli", "peele talve", "piliya"],
    "unable_to_feed": ["not feeding", "unable to suck", "doodh nahi pi raha"],
}


class TriageEngine:
    """ML-primary clinical triage with rule-based fallback and PHC/doctor assignment."""

    def __init__(self, llm=None, rules_path: str = None):
        """
        Args:
            llm: IndicLLM instance (ML-primary path).
            rules_path: Path to danger_signs.yaml (rule-based fallback).
        """
        self.llm = llm
        self.rules = {}
        if rules_path:
            self._load_rules(rules_path)

    def _load_rules(self, path: str):
        try:
            with open(path, "r") as f:
                self.rules = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load triage rules: {e}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def triage(self, visit: dict, spark=None) -> TriageResult:
        """
        Run triage on a visit record.
        1. Check hard-coded red flags (ALWAYS runs, highest priority)
        2. Try ML triage (primary)
        3. Fall back to rule-based triage if ML fails
        4. Assign PHC + doctor based on severity
        """
        # Step 1: Hard red flags — instant override
        red_flag_alerts = self._check_hard_red_flags(visit)
        if red_flag_alerts:
            result = TriageResult(
                risk_level="RED",
                alerts=red_flag_alerts,
                method="hard_red_flag",
            )
            if spark:
                result.assignment = self._assign_phc_doctor(result, visit, spark)
            return result

        # Step 2: ML triage (primary path)
        if self.llm and self.llm.is_loaded:
            ml_result = self._ml_triage(visit)
            if ml_result:
                if spark:
                    ml_result.assignment = self._assign_phc_doctor(ml_result, visit, spark)
                return ml_result

        # Step 3: Rule-based fallback
        rule_result = self._rule_based_triage(visit)
        if spark:
            rule_result.assignment = self._assign_phc_doctor(rule_result, visit, spark)
        return rule_result

    # ------------------------------------------------------------------
    # Hard-coded red flags
    # ------------------------------------------------------------------

    def _check_hard_red_flags(self, visit: dict) -> list:
        """Check for absolute red flags that bypass all classification."""
        text = " ".join([
            visit.get("clinical_notes", ""),
            " ".join(visit.get("danger_signs_observed", [])),
        ]).lower()

        alerts = []
        for flag_name, keywords in HARD_RED_FLAGS.items():
            for kw in keywords:
                if kw in text:
                    alerts.append(TriageAlert(
                        sign_name=flag_name,
                        severity="critical",
                        action="EMERGENCY REFERRAL. Transport to nearest FRU immediately.",
                        urgency_hours=0,
                        confidence=1.0,
                    ))
                    break
        return alerts

    # ------------------------------------------------------------------
    # ML triage (primary)
    # ------------------------------------------------------------------

    def _ml_triage(self, visit: dict) -> Optional[TriageResult]:
        """Use Indic LLM for clinical assessment."""
        try:
            visit_data = {
                "bp_systolic": visit.get("bp_systolic"),
                "bp_diastolic": visit.get("bp_diastolic"),
                "hemoglobin_g_dl": visit.get("hemoglobin_g_dl"),
                "temperature_c": visit.get("temperature_c"),
                "gestational_age_weeks": visit.get("gestational_age_weeks"),
                "weight_kg": visit.get("weight_kg"),
                "clinical_notes": visit.get("clinical_notes", ""),
                "danger_signs_observed": visit.get("danger_signs_observed", []),
            }
            assessment = self.llm.generate_clinical_assessment(visit_data)
            if not assessment:
                return None

            risk = assessment.get("risk_level", "YELLOW").upper()
            confidence = float(assessment.get("confidence", 0.5))

            # Confidence gating: if below threshold, escalate to RED
            if confidence < 0.85:
                logger.warning(
                    f"ML confidence {confidence:.2f} below threshold. Escalating to RED."
                )
                risk = "RED"
                alerts = [TriageAlert(
                    sign_name="low_confidence_escalation",
                    severity="high",
                    action="ML confidence low. Please consult a doctor immediately.",
                    urgency_hours=4,
                    confidence=confidence,
                )]
            else:
                danger_signs = assessment.get("danger_signs", [])
                action = assessment.get("recommended_action", "Monitor and follow up.")
                urgency = int(assessment.get("urgency_hours", 24))
                alerts = [TriageAlert(
                    sign_name=sign,
                    severity="critical" if risk == "RED" else "moderate",
                    action=action,
                    urgency_hours=urgency,
                    confidence=confidence,
                ) for sign in danger_signs] if danger_signs else [TriageAlert(
                    sign_name="ml_assessment",
                    severity="low" if risk == "GREEN" else "moderate",
                    action=action,
                    urgency_hours=urgency,
                    confidence=confidence,
                )]

            return TriageResult(risk_level=risk, alerts=alerts, method="ml")

        except Exception as e:
            logger.warning(f"ML triage failed: {e}. Falling back to rules.")
            return None

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_triage(self, visit: dict) -> TriageResult:
        """Deterministic triage from vital thresholds and keyword matching."""
        alerts = []

        # Check vital sign thresholds
        bp_s = visit.get("bp_systolic", 0)
        bp_d = visit.get("bp_diastolic", 0)
        hb = visit.get("hemoglobin_g_dl", 12.0)
        temp = visit.get("temperature_c", 36.8)

        if bp_s >= 160 or bp_d >= 110:
            alerts.append(TriageAlert(
                "severe_hypertension", "critical",
                "Severe hypertension. Refer to FRU immediately.", 0))
        elif bp_s >= 140 or bp_d >= 90:
            alerts.append(TriageAlert(
                "high_bp_preeclampsia", "high",
                "Elevated BP. Refer to PHC within 4 hours. Monitor for pre-eclampsia.", 4))

        if hb < 7.0:
            alerts.append(TriageAlert(
                "severe_anemia", "high",
                "Severe anemia. Refer to PHC for blood transfusion evaluation.", 4))
        elif hb < 9.0:
            alerts.append(TriageAlert(
                "moderate_anemia", "moderate",
                "Moderate anemia. Ensure IFA tablets. Counsel on iron-rich diet.", 48))

        if temp >= 38.5:
            alerts.append(TriageAlert(
                "high_fever", "high",
                "High fever. Refer to PHC within 4 hours. Suspect infection.", 4))

        # Keyword matching from YAML rules
        text = " ".join([
            visit.get("clinical_notes", ""),
            " ".join(visit.get("danger_signs_observed", [])),
        ]).lower()

        for category in ("maternal", "neonatal"):
            for rule in self.rules.get(category, []):
                matched = False
                for kw in rule.get("keywords_en", []) + rule.get("keywords_hi", []):
                    if kw.lower() in text:
                        matched = True
                        break
                if matched:
                    # Avoid duplicating alerts already added by vital checks
                    if not any(a.sign_name == rule["name"] for a in alerts):
                        alerts.append(TriageAlert(
                            sign_name=rule["name"],
                            severity=rule["severity"],
                            action=rule["action"],
                            urgency_hours=rule.get("urgency_hours", 24),
                        ))

        # Determine overall risk level
        if any(a.severity == "critical" for a in alerts):
            risk = "RED"
        elif any(a.severity == "high" for a in alerts):
            risk = "YELLOW"
        elif any(a.severity == "moderate" for a in alerts):
            risk = "YELLOW"
        else:
            risk = "GREEN"

        if not alerts:
            alerts = [TriageAlert(
                "normal", "low", "No danger signs detected. Continue routine monitoring.", 168)]

        return TriageResult(risk_level=risk, alerts=alerts, method="rule_fallback")

    # ------------------------------------------------------------------
    # PHC + Doctor assignment
    # ------------------------------------------------------------------

    def _assign_phc_doctor(self, triage_result: TriageResult, visit: dict,
                           spark) -> AssignmentResult:
        """
        Assign nearest appropriate PHC and available doctor based on triage severity.
        RED → FRU (First Referral Unit) with surgery/blood bank + Ob/Gyn
        YELLOW → Nearest PHC/CHC + available doctor
        GREEN → No assignment needed (routine monitoring)
        """
        if triage_result.risk_level == "GREEN":
            return AssignmentResult()

        patient_id = visit.get("patient_id", "")
        # Get patient's nearest PHC
        patient_phc = visit.get("nearest_phc_id", "")
        village = visit.get("village", "")

        try:
            if triage_result.risk_level == "RED":
                # Need FRU with surgery + blood bank
                facilities = spark.sql(f"""
                    SELECT * FROM hive_metastore.asha_copilot.phc_facilities
                    WHERE type = 'FRU' AND has_surgery = true AND has_blood_bank = true
                    ORDER BY phc_id
                    LIMIT 1
                """).collect()

                if not facilities:
                    # Fall back to any CHC
                    facilities = spark.sql("""
                        SELECT * FROM hive_metastore.asha_copilot.phc_facilities
                        WHERE type IN ('CHC', 'FRU')
                        ORDER BY phc_id LIMIT 1
                    """).collect()
            else:
                # YELLOW — nearest PHC or CHC
                facilities = spark.sql(f"""
                    SELECT * FROM hive_metastore.asha_copilot.phc_facilities
                    WHERE phc_id = '{patient_phc}'
                    LIMIT 1
                """).collect()

                if not facilities:
                    facilities = spark.sql("""
                        SELECT * FROM hive_metastore.asha_copilot.phc_facilities
                        WHERE type IN ('PHC', 'CHC')
                        ORDER BY phc_id LIMIT 1
                    """).collect()

            if not facilities:
                return AssignmentResult()

            fac = facilities[0]

            # Find available doctor at that facility
            today_day = date.today().strftime("%a")
            preferred_spec = "'Obstetrics & Gynecology', 'Obstetrics'" if triage_result.risk_level == "RED" else "'MBBS', 'Obstetrics'"
            doctors = spark.sql(f"""
                SELECT * FROM hive_metastore.asha_copilot.doctors
                WHERE phc_id = '{fac['phc_id']}'
                AND array_contains(available_days, '{today_day}')
                ORDER BY
                    CASE WHEN specialization IN ({preferred_spec}) THEN 0 ELSE 1 END,
                    doctor_id
                LIMIT 1
            """).collect()

            # If no doctor available today, get any doctor at facility
            if not doctors:
                doctors = spark.sql(f"""
                    SELECT * FROM hive_metastore.asha_copilot.doctors
                    WHERE phc_id = '{fac['phc_id']}'
                    LIMIT 1
                """).collect()

            doc = doctors[0] if doctors else None

            return AssignmentResult(
                phc_id=fac["phc_id"],
                phc_name=fac["name"],
                phc_type=fac["type"],
                doctor_id=doc["doctor_id"] if doc else "",
                doctor_name=doc["name"] if doc else "On-call doctor",
                doctor_phone=doc["phone"] if doc else "",
                doctor_specialization=doc["specialization"] if doc else "",
            )

        except Exception as e:
            logger.warning(f"PHC/doctor assignment failed: {e}")
            return AssignmentResult()
