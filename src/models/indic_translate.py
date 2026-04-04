"""
IndicTrans2 translation wrapper.

Primary: Local ONNX model (indictrans2-indic-en-dist-200M)
Fallback: AI4Bharat / Sarvam hosted API
"""

import logging
import os

logger = logging.getLogger(__name__)


class IndicTranslator:
    """Hindi ↔ English translator using IndicTrans2."""

    def __init__(self, model_dir: str = None):
        self._model = None
        self._tokenizer = None
        self._model_dir = model_dir
        self._load_model()

    def _load_model(self):
        """Try loading IndicTrans2 ONNX model locally."""
        if not self._model_dir or not os.path.exists(self._model_dir):
            logger.warning(
                f"IndicTrans2 model not found at {self._model_dir}. Using API fallback."
            )
            return
        try:
            import onnxruntime as ort
            model_path = os.path.join(self._model_dir, "model.onnx")
            if os.path.exists(model_path):
                self._model = ort.InferenceSession(model_path)
                logger.info("Loaded IndicTrans2 ONNX model.")
        except Exception as e:
            logger.warning(f"Failed to load IndicTrans2 ONNX: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def to_english(self, text: str, src_lang: str = "hi") -> str:
        """Translate from Indic language to English."""
        if not text or not text.strip():
            return text
        if src_lang == "en":
            return text
        # Try local model first
        if self._model:
            return self._translate_local(text, src_lang, "en")
        # API fallback
        return self._translate_api(text, src_lang, "en")

    def from_english(self, text: str, tgt_lang: str = "hi") -> str:
        """Translate from English to Indic language."""
        if not text or not text.strip():
            return text
        if tgt_lang == "en":
            return text
        if self._model:
            return self._translate_local(text, "en", tgt_lang)
        return self._translate_api(text, "en", tgt_lang)

    def _translate_local(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate using local ONNX model."""
        try:
            # Simplified ONNX inference — actual implementation depends on
            # the specific IndicTrans2 ONNX export format
            inputs = {"input_text": [text], "src_lang": [src_lang], "tgt_lang": [tgt_lang]}
            result = self._model.run(None, inputs)
            return result[0][0] if result else text
        except Exception as e:
            logger.warning(f"Local translation failed: {e}. Trying API.")
            return self._translate_api(text, src_lang, tgt_lang)

    def _translate_api(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate using AI4Bharat / Sarvam API."""
        try:
            import requests

            # Try AI4Bharat IndicTrans2 on HuggingFace
            api_key = os.environ.get("HF_TOKEN", "")
            lang_map = {"hi": "hin_Deva", "en": "eng_Latn", "ta": "tam_Taml",
                        "bn": "ben_Beng", "te": "tel_Telu", "mr": "mar_Deva"}
            src = lang_map.get(src_lang, src_lang)
            tgt = lang_map.get(tgt_lang, tgt_lang)

            resp = requests.post(
                "https://api-inference.huggingface.co/models/ai4bharat/indictrans2-indic-en-dist-200M",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"inputs": text, "parameters": {"src_lang": src, "tgt_lang": tgt}},
                timeout=15,
            )
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and result:
                    return result[0].get("translation_text", text)
                return result.get("translation_text", text)
        except Exception as e:
            logger.warning(f"Translation API failed: {e}")

        # Last resort: return original text (better than crashing)
        logger.warning("All translation methods failed. Returning original text.")
        return text

    def detect_language(self, text: str) -> str:
        """Simple heuristic to detect Hindi vs English."""
        hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha == 0:
            return "en"
        return "hi" if hindi_chars / total_alpha > 0.3 else "en"
