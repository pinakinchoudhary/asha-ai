"""
IndicTrans2 translation wrapper.

Primary: Sarvam AI Mayura translation API (Hindi ↔ English)
Fallback: AI4Bharat IndicTrans2 via HuggingFace Inference API
"""

import logging
import os

logger = logging.getLogger(__name__)

SARVAM_API_BASE = "https://api.sarvam.ai"

# Sarvam language codes
_SARVAM_LANG = {
    "hi": "hi-IN",
    "en": "en-IN",
    "ta": "ta-IN",
    "bn": "bn-IN",
    "te": "te-IN",
    "mr": "mr-IN",
}

# HuggingFace IndicTrans2 language codes
_HF_LANG = {
    "hi": "hin_Deva",
    "en": "eng_Latn",
    "ta": "tam_Taml",
    "bn": "ben_Beng",
    "te": "tel_Telu",
    "mr": "mar_Deva",
}


class IndicTranslator:
    """Hindi ↔ English translator using Sarvam Mayura API."""

    def __init__(self, model_dir: str = None):
        # model_dir is ignored — Sarvam API is used instead of local ONNX
        self._sarvam_key = os.environ.get("SARVAM_API_KEY", "")
        self._hf_key = os.environ.get("HF_TOKEN", "")
        if not self._sarvam_key:
            logger.warning(
                "SARVAM_API_KEY not set. Translation will use HF API fallback."
            )

    @property
    def is_loaded(self) -> bool:
        return bool(self._sarvam_key or self._hf_key)

    def to_english(self, text: str, src_lang: str = "hi") -> str:
        """Translate from Indic language to English."""
        if not text or not text.strip():
            return text
        if src_lang == "en":
            return text
        return self._translate(text, src_lang, "en")

    def from_english(self, text: str, tgt_lang: str = "hi") -> str:
        """Translate from English to Indic language."""
        if not text or not text.strip():
            return text
        if tgt_lang == "en":
            return text
        return self._translate(text, "en", tgt_lang)

    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate via Sarvam Mayura API, fallback to HF."""
        if self._sarvam_key:
            result = self._translate_sarvam(text, src_lang, tgt_lang)
            if result and result != text:
                return result
        return self._translate_hf(text, src_lang, tgt_lang)

    def _translate_sarvam(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate using Sarvam AI Mayura API."""
        try:
            import requests
            src = _SARVAM_LANG.get(src_lang, f"{src_lang}-IN")
            tgt = _SARVAM_LANG.get(tgt_lang, f"{tgt_lang}-IN")
            resp = requests.post(
                f"{SARVAM_API_BASE}/translate",
                headers={
                    "api-subscription-key": self._sarvam_key,
                    "Content-Type": "application/json",
                },
                json={
                    "input": text,
                    "source_language_code": src,
                    "target_language_code": tgt,
                    "speaker_gender": "Female",
                    "mode": "formal",
                    "model": "mayura:v1",
                    "enable_preprocessing": False,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json().get("translated_text", text)
            logger.warning(f"Sarvam translate returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Sarvam translate failed: {e}")
        return ""

    def _translate_hf(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate using AI4Bharat IndicTrans2 on HuggingFace (fallback)."""
        if not self._hf_key:
            logger.warning("No HF_TOKEN. Returning original text.")
            return text
        try:
            import requests
            src = _HF_LANG.get(src_lang, src_lang)
            tgt = _HF_LANG.get(tgt_lang, tgt_lang)
            resp = requests.post(
                "https://api-inference.huggingface.co/models/ai4bharat/indictrans2-indic-en-dist-200M",
                headers={"Authorization": f"Bearer {self._hf_key}"},
                json={"inputs": text, "parameters": {"src_lang": src, "tgt_lang": tgt}},
                timeout=15,
            )
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and result:
                    return result[0].get("translation_text", text)
                return result.get("translation_text", text)
        except Exception as e:
            logger.warning(f"HF translation fallback failed: {e}")
        logger.warning("All translation methods failed. Returning original text.")
        return text

    def detect_language(self, text: str) -> str:
        """Simple heuristic to detect Hindi vs English."""
        hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha == 0:
            return "en"
        return "hi" if hindi_chars / total_alpha > 0.3 else "en"
