"""
Unified Indic LLM interface.

Primary: Airavata 7B GGUF (Hindi instruction-tuned) via llama-cpp-python
Lightweight: Param-1 2.9B GGUF for intent classification / entity extraction
Fallback: Hugging Face Inference API
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


class IndicLLM:
    """Wrapper around quantized Indic LLMs with graceful degradation."""

    def __init__(self, model_path: str = None, n_ctx: int = 2048, n_threads: int = 4):
        self._model = None
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._load_model()

    def _load_model(self):
        """Try loading GGUF model via llama-cpp-python. Fail gracefully."""
        if not self._model_path or not os.path.exists(self._model_path):
            logger.warning(f"Model not found at {self._model_path}. Using API fallback.")
            return
        try:
            from llama_cpp import Llama
            self._model = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                verbose=False,
            )
            logger.info(f"Loaded GGUF model from {self._model_path}")
        except Exception as e:
            logger.warning(f"Failed to load GGUF model: {e}. Using API fallback.")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3,
                 stop: list = None) -> str:
        """Generate text from prompt. Falls back to API if local model unavailable."""
        if self._model:
            return self._generate_local(prompt, max_tokens, temperature, stop)
        return self._generate_api_fallback(prompt, max_tokens, temperature)

    def _generate_local(self, prompt: str, max_tokens: int, temperature: float,
                        stop: list = None) -> str:
        output = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or ["</s>", "\n\n\n"],
        )
        return output["choices"][0]["text"].strip()

    def _generate_api_fallback(self, prompt: str, max_tokens: int,
                               temperature: float) -> str:
        """Call Hugging Face Inference API as fallback."""
        try:
            import requests
            api_key = os.environ.get("HF_TOKEN", "")
            model_id = "ai4bharat/Airavata"
            resp = requests.post(
                f"https://api-inference.huggingface.co/models/{model_id}",
                headers={"Authorization": f"Bearer {api_key}"},
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
            logger.warning(f"API fallback returned {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"API fallback failed: {e}")
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
            "Intent:"
        )
        result = self.generate(prompt, max_tokens=20, temperature=0.1)
        result = result.strip().lower().replace('"', '').replace("'", "")
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
                "Extract patient information from the following text. Return ONLY valid JSON.\n\n"
                f"Text: {text}\n\n"
                "Extract these fields (use null if not mentioned):\n"
                '{"name": str, "age": int, "village": str, "husband_name": str, '
                '"bp_systolic": int, "bp_diastolic": int, "weight_kg": float, '
                '"hemoglobin": float, "temperature": float, "symptoms": [str], '
                '"gravida": int, "para": int, "lmp_date": str, '
                '"bpl_status": bool, "caste_category": str, '
                '"aadhaar": bool, "bank_account": bool}\n\n'
                "JSON:"
            )
        elif intent == "scheme_query":
            prompt = (
                "Extract patient details for scheme eligibility from this text. "
                "Return ONLY valid JSON.\n\n"
                f"Text: {text}\n\n"
                '{"name": str, "age": int, "parity": int, "bpl_status": bool, '
                '"caste_category": str, "state": str, "institutional_delivery": bool}\n\n'
                "JSON:"
            )
        else:
            prompt = (
                "Extract the key question or topic from this text. Return ONLY valid JSON.\n\n"
                f"Text: {text}\n\n"
                '{"query": str, "patient_name": str, "child_name": str}\n\n'
                "JSON:"
            )

        result = self.generate(prompt, max_tokens=300, temperature=0.1)
        try:
            # Find JSON in the response
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(result[start:end])
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Failed to parse entities from LLM output: {result}")
        return {}

    def generate_clinical_assessment(self, visit_data: dict) -> str:
        """Generate clinical assessment from visit vitals and symptoms."""
        prompt = (
            "You are a clinical triage assistant for Indian rural healthcare.\n"
            "Assess the following patient visit and classify risk.\n\n"
            f"Patient vitals and notes:\n{json.dumps(visit_data, indent=2)}\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"risk_level": "GREEN|YELLOW|RED", '
            '"danger_signs": ["list of detected signs"], '
            '"recommended_action": "specific action", '
            '"urgency_hours": int, '
            '"confidence": float between 0 and 1}\n\n'
            "Assessment JSON:"
        )
        result = self.generate(prompt, max_tokens=300, temperature=0.2)
        try:
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(result[start:end])
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Failed to parse clinical assessment: {result}")
        return None
