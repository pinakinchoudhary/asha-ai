"""
Unified Indic LLM interface.

Primary: Sarvam AI API (sarvam-m — multilingual, strong Hindi support)
Fallback: Hugging Face Inference API
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

SARVAM_API_BASE = "https://api.sarvam.ai"
SARVAM_LLM_MODEL = "sarvam-m"


class IndicLLM:
    """Wrapper around Sarvam AI API with graceful degradation to HuggingFace."""

    def __init__(self, model_path: str = None, n_ctx: int = 2048, n_threads: int = 4):
        # model_path is ignored — Sarvam API is used instead of local GGUF
        self._sarvam_key = os.environ.get("SARVAM_API_KEY", "")
        self._hf_key = os.environ.get("HF_TOKEN", "")
        if not self._sarvam_key:
            logger.warning(
                "SARVAM_API_KEY not set. Set it via os.environ or Databricks cluster config. "
                "Falling back to HuggingFace Inference API."
            )

    @property
    def is_loaded(self) -> bool:
        """True if any API key is available."""
        return bool(self._sarvam_key or self._hf_key)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3,
                 stop: list = None) -> str:
        """Generate text from prompt using Sarvam API (falls back to HF API)."""
        if self._sarvam_key:
            result = self._generate_sarvam(prompt, max_tokens, temperature)
            if result:
                return result
        return self._generate_hf_fallback(prompt, max_tokens, temperature)

    def _generate_sarvam(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Sarvam AI chat completions endpoint (OpenAI-compatible)."""
        try:
            import requests
            resp = requests.post(
                f"{SARVAM_API_BASE}/v1/chat/completions",
                headers={
                    "api-subscription-key": self._sarvam_key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": SARVAM_LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=45,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            logger.warning(f"Sarvam API returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Sarvam API call failed: {e}")
        return ""

    def _generate_hf_fallback(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call HuggingFace Inference API as fallback."""
        if not self._hf_key:
            logger.warning("No HF_TOKEN set. Cannot use HF fallback.")
            return ""
        try:
            import requests
            resp = requests.post(
                f"https://api-inference.huggingface.co/models/ai4bharat/Airavata",
                headers={"Authorization": f"Bearer {self._hf_key}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                    },
                },
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()[0]["generated_text"].replace(prompt, "").strip()
            logger.warning(f"HF API returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"HF API fallback failed: {e}")
        return ""

    def classify_intent(self, text: str) -> str:
        """Classify user intent from ASHA's speech (translated to English)."""
        prompt = (
            "You are a healthcare assistant classifier. Classify the following input "
            "into exactly ONE of these intents:\n"
            "- symptom_report: Patient is reporting symptoms or health complaints\n"
            "- add_patient: Registering a new patient\n"
            "- update_record: Updating an existing patient's record or vitals\n"
            "- log_visit: Recording a home visit with vitals and observations\n"
            "- scheme_query: Asking about government welfare schemes or eligibility\n"
            "- immunization_check: Asking about vaccination schedule or status\n"
            "- protocol_question: Asking about clinical protocols or guidelines\n\n"
            f"Input: {text}\n\n"
            "Respond with ONLY the intent name (one word):\nIntent:"
        )
        result = self.generate(prompt, max_tokens=20, temperature=0.1)
        result = result.strip().lower().replace('"', '').replace("'", "").split()[0] if result.strip() else ""
        valid_intents = [
            "symptom_report", "add_patient", "update_record", "log_visit",
            "scheme_query", "immunization_check", "protocol_question",
        ]
        for intent in valid_intents:
            if intent in result:
                return intent
        return "protocol_question"  # safe default

    def extract_entities(self, text: str, intent: str) -> dict:
        """Extract structured entities from ASHA's speech based on intent."""
        if intent in ("add_patient", "log_visit", "update_record"):
            prompt = (
                "Extract patient information from the following text. Return ONLY valid JSON with no extra text.\n\n"
                f"Text: {text}\n\n"
                "Extract these fields (use null if not mentioned):\n"
                '{"name": "string or null", "age": number_or_null, "village": "string or null", '
                '"husband_name": "string or null", '
                '"bp_systolic": number_or_null, "bp_diastolic": number_or_null, "weight_kg": number_or_null, '
                '"hemoglobin": number_or_null, "temperature": number_or_null, "symptoms": ["list"],  '
                '"gravida": number_or_null, "para": number_or_null, '
                '"bpl_status": true/false/null, "caste_category": "string or null", '
                '"aadhaar": true/false/null, "bank_account": true/false/null}\n\n'
                "JSON:"
            )
        elif intent == "scheme_query":
            prompt = (
                "Extract patient details for scheme eligibility from this text. "
                "Return ONLY valid JSON with no extra text.\n\n"
                f"Text: {text}\n\n"
                '{"name": "string or null", "age": number_or_null, "parity": number_or_null, '
                '"bpl_status": true/false/null, '
                '"caste_category": "string or null", "state": "string or null", '
                '"institutional_delivery": true/false/null}\n\n'
                "JSON:"
            )
        else:
            prompt = (
                "Extract the key question or topic from this text. Return ONLY valid JSON with no extra text.\n\n"
                f"Text: {text}\n\n"
                '{"query": "string", "patient_name": "string or null", "child_name": "string or null"}\n\n'
                "JSON:"
            )

        result = self.generate(prompt, max_tokens=300, temperature=0.1)
        try:
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(result[start:end])
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Failed to parse entities from LLM output: {result[:200]}")
        return {}

    def generate_clinical_assessment(self, visit_data: dict) -> dict:
        """Generate clinical assessment from visit vitals and symptoms."""
        prompt = (
            "You are a clinical triage assistant for Indian rural healthcare.\n"
            "Assess the following patient visit and classify risk.\n\n"
            f"Patient vitals and notes:\n{json.dumps(visit_data, indent=2)}\n\n"
            "Respond ONLY with valid JSON and nothing else:\n"
            '{"risk_level": "GREEN or YELLOW or RED", '
            '"danger_signs": ["list of detected signs"], '
            '"recommended_action": "specific action", '
            '"urgency_hours": number, '
            '"confidence": number_between_0_and_1}\n\n'
            "Assessment JSON:"
        )
        result = self.generate(prompt, max_tokens=300, temperature=0.2)
        try:
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(result[start:end])
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Failed to parse clinical assessment: {result[:200]}")
        return None
