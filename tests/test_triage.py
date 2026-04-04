"""
Tests for the clinical triage engine (rule-based fallback path).
Runs locally with pytest — no Spark required.
"""

import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.triage_engine import TriageEngine

RULES_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "danger_signs.yaml")


@pytest.fixture
def engine():
    """Triage engine with no LLM (rule-based fallback)."""
    return TriageEngine(llm=None, rules_path=RULES_PATH)


# ---------------------------------------------------------------------------
# Vital-based rules
# ---------------------------------------------------------------------------

def test_severe_hypertension_triggers_red(engine):
    visit = {"bp_systolic": 165, "bp_diastolic": 112, "hemoglobin_g_dl": 11.0,
             "temperature_c": 37.0, "clinical_notes": "", "danger_signs_observed": []}
    result = engine.triage(visit)
    assert result.risk_level == "RED"
    assert any("hypertension" in a.sign_name for a in result.alerts)


def test_high_bp_triggers_yellow(engine):
    visit = {"bp_systolic": 145, "bp_diastolic": 92, "hemoglobin_g_dl": 11.0,
             "temperature_c": 37.0, "clinical_notes": "", "danger_signs_observed": []}
    result = engine.triage(visit)
    assert result.risk_level in ("RED", "YELLOW")
    assert any("bp" in a.sign_name or "preeclampsia" in a.sign_name for a in result.alerts)


def test_severe_anemia_triggers_red_or_yellow(engine):
    visit = {"bp_systolic": 110, "bp_diastolic": 70, "hemoglobin_g_dl": 6.0,
             "temperature_c": 37.0, "clinical_notes": "", "danger_signs_observed": []}
    result = engine.triage(visit)
    assert result.risk_level in ("RED", "YELLOW")
    assert any("anemia" in a.sign_name for a in result.alerts)


def test_moderate_anemia_triggers_yellow(engine):
    visit = {"bp_systolic": 115, "bp_diastolic": 75, "hemoglobin_g_dl": 8.0,
             "temperature_c": 37.0, "clinical_notes": "", "danger_signs_observed": []}
    result = engine.triage(visit)
    assert result.risk_level == "YELLOW"
    assert any("anemia" in a.sign_name for a in result.alerts)


def test_high_fever_triggers_yellow(engine):
    visit = {"bp_systolic": 120, "bp_diastolic": 80, "hemoglobin_g_dl": 11.0,
             "temperature_c": 39.5, "clinical_notes": "", "danger_signs_observed": []}
    result = engine.triage(visit)
    assert result.risk_level == "YELLOW"
    assert any("fever" in a.sign_name for a in result.alerts)


def test_normal_visit_is_green(engine):
    visit = {"bp_systolic": 118, "bp_diastolic": 75, "hemoglobin_g_dl": 12.0,
             "temperature_c": 36.8, "clinical_notes": "Sab theek hai, koi problem nahi.",
             "danger_signs_observed": []}
    result = engine.triage(visit)
    assert result.risk_level == "GREEN"


# ---------------------------------------------------------------------------
# Hard-coded red flags — ALWAYS override
# ---------------------------------------------------------------------------

def test_convulsions_keyword_forces_red(engine):
    visit = {"bp_systolic": 110, "bp_diastolic": 70, "hemoglobin_g_dl": 12.0,
             "temperature_c": 37.0,
             "clinical_notes": "Patient ko daurey aa rahe hain, convulsions ho rahi hain",
             "danger_signs_observed": []}
    result = engine.triage(visit)
    assert result.risk_level == "RED"
    assert result.method == "hard_red_flag"


def test_severe_bleeding_keyword_forces_red(engine):
    visit = {"bp_systolic": 100, "bp_diastolic": 65, "hemoglobin_g_dl": 10.0,
             "temperature_c": 37.0,
             "clinical_notes": "bahut bleeding ho rahi hai, severe bleeding",
             "danger_signs_observed": []}
    result = engine.triage(visit)
    assert result.risk_level == "RED"
    assert result.method == "hard_red_flag"


def test_chest_indrawing_forces_red(engine):
    visit = {"bp_systolic": 110, "bp_diastolic": 70, "hemoglobin_g_dl": 12.0,
             "temperature_c": 37.5,
             "clinical_notes": "Bachche ki chhati dhans rahi hai, chest indrawing observed",
             "danger_signs_observed": []}
    result = engine.triage(visit)
    assert result.risk_level == "RED"
    assert result.method == "hard_red_flag"


def test_yellow_palms_forces_red(engine):
    visit = {"bp_systolic": 110, "bp_diastolic": 70, "hemoglobin_g_dl": 12.0,
             "temperature_c": 37.0,
             "clinical_notes": "Baby ki peeli hatheli aur peele talve hain. Yellow palms.",
             "danger_signs_observed": []}
    result = engine.triage(visit)
    assert result.risk_level == "RED"
    assert result.method == "hard_red_flag"


def test_unable_to_feed_forces_red(engine):
    visit = {"bp_systolic": 110, "bp_diastolic": 70, "hemoglobin_g_dl": 12.0,
             "temperature_c": 37.0,
             "clinical_notes": "Bachcha doodh nahi pi raha, not feeding since morning",
             "danger_signs_observed": []}
    result = engine.triage(visit)
    assert result.risk_level == "RED"
    assert result.method == "hard_red_flag"


# ---------------------------------------------------------------------------
# Keyword matching from YAML rules
# ---------------------------------------------------------------------------

def test_vaginal_bleeding_from_danger_signs_array(engine):
    visit = {"bp_systolic": 120, "bp_diastolic": 80, "hemoglobin_g_dl": 10.0,
             "temperature_c": 37.0, "clinical_notes": "",
             "danger_signs_observed": ["vaginal bleeding"]}
    result = engine.triage(visit)
    assert result.risk_level == "RED"


def test_decreased_fetal_movement_keyword(engine):
    # "hil nahi raha" also matches neonatal_sepsis (critical) → RED is correct
    visit = {"bp_systolic": 120, "bp_diastolic": 80, "hemoglobin_g_dl": 11.0,
             "temperature_c": 37.0,
             "clinical_notes": "Bachcha hil nahi raha, decreased fetal movement",
             "danger_signs_observed": []}
    result = engine.triage(visit)
    assert result.risk_level in ("RED", "YELLOW")
    assert any("fetal_movement" in a.sign_name or "sepsis" in a.sign_name
               for a in result.alerts)


# ---------------------------------------------------------------------------
# Method tracking
# ---------------------------------------------------------------------------

def test_rule_fallback_method_when_no_llm(engine):
    visit = {"bp_systolic": 120, "bp_diastolic": 80, "hemoglobin_g_dl": 11.0,
             "temperature_c": 37.0, "clinical_notes": "Routine visit",
             "danger_signs_observed": []}
    result = engine.triage(visit)
    assert result.method == "rule_fallback"


def test_triage_result_serializable(engine):
    visit = {"bp_systolic": 160, "bp_diastolic": 100, "hemoglobin_g_dl": 6.5,
             "temperature_c": 39.0, "clinical_notes": "bahut beemar hai",
             "danger_signs_observed": ["severe headache"]}
    result = engine.triage(visit)
    d = result.to_dict()
    assert "risk_level" in d
    assert "alerts" in d
    assert isinstance(d["alerts"], list)
