"""
Unified Indic LLM interface.

Primary: Groq API (llama-3.3-70b-versatile — fast, strong multilingual/Hindi support)
Fallback: Hugging Face Inference Router API
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

GROQ_API_BASE = "https://api.groq.com/openai"
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"  # swap: mixtral-8x7b-32768, gemma2-9b-it
_SECRETS_SCOPE = "asha-ai"


def _get_api_key(env_var: str, secret_key: str) -> str:
    """
    Retrieve an API key.
    Priority: environment variable → Databricks secrets scope 'asha-ai'.
    Caches the secret in os.environ so subsequent calls are fast.
    """
    val = os.environ.get(env_var, "")
    if val:
        return val
    try:
        import IPython
        _ip = IPython.get_ipython()
        if _ip is not None:
            _dbutils = _ip.user_ns.get("dbutils")
            if _dbutils:
                secret = _dbutils.secrets.get(scope=_SECRETS_SCOPE, key=secret_key)
                if secret:
                    os.environ[env_var] = secret
                    return secret
    except Exception:
        pass
    return ""


class IndicLLM:
    """Wrapper around Groq API with HuggingFace fallback."""

    def __init__(self, model_path: str = None, n_ctx: int = 2048, n_threads: int = 4):
        # model_path is ignored — Groq API is used instead of local GGUF
        self._groq_key = _get_api_key("GROQ_API_KEY", "groq-api-key")
        self._hf_key = _get_api_key("HF_TOKEN", "hf-token")
        if not self._groq_key:
            logger.warning(
                "GROQ_API_KEY not set. Configure via: (1) Databricks secrets scope "
                "'asha-ai' key 'groq-api-key', or (2) cluster Environment Variables, "
                "or (3) os.environ['GROQ_API_KEY'] = '<key>'. "
                "Falling back to HuggingFace Inference Router API."
            )

    @property
    def is_loaded(self) -> bool:
        """True if any API key is available."""
        return bool(self._groq_key or self._hf_key)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3,
                 stop: list = None) -> str:
        """Generate text from prompt using Groq API (falls back to HF API)."""
        if self._groq_key:
            result = self._generate_groq(prompt, max_tokens, temperature)
            if result:
                return result
        return self._generate_hf_fallback(prompt, max_tokens, temperature)

    def _generate_groq(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call Groq chat completions endpoint (OpenAI-compatible)."""
        try:
            import requests
            resp = requests.post(
                f"{GROQ_API_BASE}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"].get("content")
                if content is not None:
                    return content.strip()
                logger.warning("Groq returned 200 but content is None")
            else:
                logger.warning(f"Groq API returned {resp.status_code}: {resp.text[:300]}")
        except Exception as e:
            logger.warning(f"Groq API call failed: {e}")
        return ""

    def _generate_hf_fallback(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Call HuggingFace Inference Router as fallback."""
        if not self._hf_key:
            logger.warning("No HF_TOKEN set. Cannot use HF fallback.")
            return ""
        try:
            import requests
            resp = requests.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._hf_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"].get("content")
                if content is not None:
                    return content.strip()
            logger.warning(f"HF Router returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"HF Router fallback failed: {e}")
        return ""

    def classify_intent(self, text: str) -> str:
        """Classify user intent from ASHA's speech (translated to English)."""
        prompt = (
            "You are a healthcare assistant classifier. Classify the following input "
            "into exactly ONE of these intents:\n"
            "- add_patient: Registering or adding a brand new patient record\n"
            "- log_visit: Recording a home visit with vitals/measurements (BP, Hb, weight, temp)\n"
            "- update_record: Updating an existing patient's personal details\n"
            "- symptom_report: Patient reporting symptoms, complaints, or danger signs\n"
            "- query_patient: Finding, showing, listing, or asking about existing patient info "
            "(e.g. 'show all patients', 'list patients', 'what is X village', 'find patient X', "
            "'mujhe patients dikhao', 'ka gaanv konsa hai')\n"
            "- scheme_query: Asking about government welfare scheme eligibility\n"
            "- immunization_check: Asking about vaccination schedule or status\n"
            "- protocol_question: Asking about clinical protocols or medical guidelines\n\n"
            f"Input: {text}\n\n"
            "Respond with ONLY the intent name (no explanation):\nIntent:"
        )
        result = self.generate(prompt, max_tokens=20, temperature=0.1)
        result = result.strip().lower().replace('"', '').replace("'", "").split()[0] if result.strip() else ""
        valid_intents = [
            "add_patient", "log_visit", "update_record", "symptom_report",
            "query_patient", "scheme_query", "immunization_check", "protocol_question",
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

    def answer_patient_query(self, question: str, patients: list) -> str:
        """Answer a specific patient question from retrieved records, in the same language as the question."""
        prompt = (
            "You are a healthcare assistant for ASHA workers in India. "
            "Answer the question below using ONLY the patient records provided. "
            "IMPORTANT: Reply in the EXACT SAME LANGUAGE as the question — if the question is in Hindi, answer in Hindi; if English, answer in English. "
            "Be brief and direct. For a single patient query, answer only what was asked (e.g. just the age, just the village). "
            "Do NOT list all patients unless the question explicitly asks for a list.\n\n"
            f"Patient records:\n{json.dumps(patients, indent=2, ensure_ascii=False)}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return self.generate(prompt, max_tokens=150, temperature=0.1)

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