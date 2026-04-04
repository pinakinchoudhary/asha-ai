"""
Welfare Scheme Eligibility Engine.

ML-assisted evaluation with rule-based fallback for PMMVY, JSY, JSSK.
"""

import json
import logging
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Low Performing States for JSY
LPS_STATES = {
    "Uttar Pradesh", "Bihar", "Madhya Pradesh", "Rajasthan", "Jharkhand",
    "Chhattisgarh", "Odisha", "Uttarakhand", "Jammu & Kashmir", "Assam",
}


@dataclass
class EligibilityResult:
    scheme_id: str
    scheme_name: str
    eligible: bool
    missing_requirements: list = field(default_factory=list)
    entitlements: list = field(default_factory=list)
    amount: int = 0
    notes: str = ""

    def to_dict(self):
        return asdict(self)


class SchemeEngine:
    """ML-assisted welfare scheme eligibility with rule-based fallback."""

    def __init__(self, llm=None):
        self.llm = llm

    def evaluate(self, patient: dict) -> list:
        """
        Evaluate all scheme eligibilities for a patient.
        Tries ML first, falls back to rules.
        """
        # Try ML-assisted evaluation
        if self.llm and self.llm.is_loaded:
            ml_result = self._ml_evaluate(patient)
            if ml_result:
                return ml_result

        # Rule-based fallback
        return self._rule_evaluate(patient)

    # ------------------------------------------------------------------
    # ML-assisted evaluation
    # ------------------------------------------------------------------

    def _ml_evaluate(self, patient: dict) -> list:
        """Use LLM to reason about scheme eligibility (handles edge cases)."""
        try:
            prompt = (
                "You are a government welfare scheme eligibility assistant for Indian "
                "maternal healthcare. Evaluate the following patient for these schemes:\n\n"
                "1. PMMVY (Pradhan Mantri Matru Vandana Yojana): Rs 5,000 for first live "
                "birth. Age >= 19, needs Aadhaar + bank account. PMMVY 2.0: Rs 6,000 "
                "single installment for second child if girl.\n"
                "2. JSY (Janani Suraksha Yojana): Cash incentive for institutional delivery. "
                "BPL/SC/ST eligible. Low Performing States: Rs 1,400 (rural), Rs 1,000 (urban). "
                "High Performing States: Rs 700 (rural), Rs 600 (urban).\n"
                "3. JSSK (Janani Shishu Suraksha Karyakram): Free cashless delivery, drugs, "
                "diagnostics, blood, diet, transport for ALL pregnant women.\n\n"
                f"Patient details:\n{json.dumps(patient, indent=2, default=str)}\n\n"
                "Return ONLY valid JSON array:\n"
                '[{"scheme_id": str, "eligible": bool, "missing": [str], '
                '"amount": int, "notes": str}]\n\n'
                "JSON:"
            )
            result = self.llm.generate(prompt, max_tokens=400, temperature=0.1)
            start = result.find("[")
            end = result.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(result[start:end])
                results = []
                scheme_names = {
                    "PMMVY": "Pradhan Mantri Matru Vandana Yojana",
                    "JSY": "Janani Suraksha Yojana",
                    "JSSK": "Janani Shishu Suraksha Karyakram",
                }
                for item in parsed:
                    sid = item.get("scheme_id", "")
                    results.append(EligibilityResult(
                        scheme_id=sid,
                        scheme_name=scheme_names.get(sid, sid),
                        eligible=item.get("eligible", False),
                        missing_requirements=item.get("missing", []),
                        amount=item.get("amount", 0),
                        notes=item.get("notes", ""),
                    ))
                return results if results else None
        except Exception as e:
            logger.warning(f"ML scheme evaluation failed: {e}")
        return None

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_evaluate(self, patient: dict) -> list:
        """Deterministic eligibility evaluation."""
        results = [
            self._check_pmmvy(patient),
            self._check_jsy(patient),
            self._check_jssk(patient),
        ]
        return results

    def _check_pmmvy(self, patient: dict) -> EligibilityResult:
        """Pradhan Mantri Matru Vandana Yojana."""
        missing = []
        age = patient.get("age", 0)
        para = patient.get("para", 0)
        has_aadhaar = patient.get("aadhaar_registered", False)
        has_bank = patient.get("bank_account", False)

        if age < 19:
            missing.append("Age must be 19 or above")
        if not has_aadhaar:
            missing.append("Aadhaar card registration")
        if not has_bank:
            missing.append("Bank account (for direct benefit transfer)")

        # PMMVY 2.0 logic
        if para == 0:
            amount = 5000  # first child
            notes = "First pregnancy: Rs 5,000 in 3 installments (Rs 3,000 + Rs 1,000 + Rs 1,000)"
        elif para == 1:
            amount = 6000  # second child if girl — we assume eligibility
            notes = "PMMVY 2.0: Rs 6,000 single installment (if second child is a girl)"
        else:
            missing.append("PMMVY covers only first two pregnancies")
            amount = 0
            notes = "Not eligible: PMMVY is for first and second pregnancy only"

        eligible = len(missing) == 0 and amount > 0
        entitlements = []
        if eligible:
            entitlements = [
                "Cash transfer to bank account",
                "Partial wage compensation for pregnancy/childbirth",
            ]

        return EligibilityResult(
            scheme_id="PMMVY",
            scheme_name="Pradhan Mantri Matru Vandana Yojana",
            eligible=eligible,
            missing_requirements=missing,
            entitlements=entitlements,
            amount=amount,
            notes=notes,
        )

    def _check_jsy(self, patient: dict) -> EligibilityResult:
        """Janani Suraksha Yojana."""
        missing = []
        state = patient.get("state", "")
        bpl = patient.get("bpl_status", False)
        caste = patient.get("caste_category", "General")
        age = patient.get("age", 0)
        is_lps = state in LPS_STATES

        if age < 19:
            missing.append("Age must be 19 or above")

        # In LPS: all pregnant women in govt facilities
        # In HPS: BPL / SC / ST only
        if not is_lps and caste == "General" and not bpl:
            missing.append("In High Performing States, BPL/SC/ST status required")

        if is_lps:
            amount = 1400  # rural (assume rural for demo)
            notes = f"{state} is a Low Performing State. Rural benefit: Rs 1,400"
        else:
            amount = 700
            notes = f"{state} is a High Performing State. Rural benefit: Rs 700"

        eligible = len(missing) == 0
        entitlements = []
        if eligible:
            entitlements = [
                f"Cash incentive: Rs {amount} for institutional delivery",
                "ASHA incentive: Rs 600 for facilitating delivery",
            ]

        return EligibilityResult(
            scheme_id="JSY",
            scheme_name="Janani Suraksha Yojana",
            eligible=eligible,
            missing_requirements=missing,
            entitlements=entitlements,
            amount=amount,
            notes=notes,
        )

    def _check_jssk(self, patient: dict) -> EligibilityResult:
        """Janani Shishu Suraksha Karyakram — universal entitlement."""
        return EligibilityResult(
            scheme_id="JSSK",
            scheme_name="Janani Shishu Suraksha Karyakram",
            eligible=True,
            missing_requirements=[],
            entitlements=[
                "Free and cashless normal delivery",
                "Free and cashless C-section",
                "Free drugs and consumables",
                "Free diagnostics (blood, urine tests, ultrasound)",
                "Free blood provision",
                "Free diet during hospital stay",
                "Free transport (home to facility and back)",
                "Free treatment for sick infants up to 1 year",
            ],
            amount=0,
            notes="Universal entitlement for ALL pregnant women and sick newborns. No eligibility criteria.",
        )
