"""
Indic Automatic Speech Recognition (ASR).

Primary: Sarvam AI Saarika ASR API
Fallback: HuggingFace IndicWhisper API, then text passthrough for demo
"""

import logging
import os

logger = logging.getLogger(__name__)

SARVAM_API_BASE = "https://api.sarvam.ai"
_SECRETS_SCOPE = "asha-ai"


def _get_api_key(env_var: str, secret_key: str) -> str:
    """Retrieve API key: env var first, then Databricks secrets."""
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

_SARVAM_LANG = {
    "hi": "hi-IN",
    "en": "en-IN",
}


class IndicASR:
    """Speech-to-text for Hindi and English."""

    def __init__(self):
        self._sarvam_key = _get_api_key("SARVAM_API_KEY", "sarvam-api-key")
        self._hf_key = _get_api_key("HF_TOKEN", "hf-token")

    def transcribe(self, audio_input, language: str = "hi") -> str:
        """
        Transcribe audio to text.

        Args:
            audio_input: File path (str), bytes, or text string (passthrough for demo).
            language: Source language code ("hi" or "en").

        Returns:
            Transcribed text string.
        """
        # If input is already text, pass through (demo mode)
        if isinstance(audio_input, str) and not os.path.exists(audio_input):
            logger.info("ASR: Text input detected (demo mode passthrough).")
            return audio_input

        # Read audio file if path provided
        if isinstance(audio_input, str) and os.path.exists(audio_input):
            with open(audio_input, "rb") as f:
                audio_bytes = f.read()
            filename = os.path.basename(audio_input)
        elif isinstance(audio_input, bytes):
            audio_bytes = audio_input
            filename = "audio.wav"
        else:
            logger.warning("ASR: Invalid input type. Returning empty string.")
            return ""

        # Try Sarvam Saarika first
        if self._sarvam_key:
            result = self._transcribe_sarvam(audio_bytes, filename, language)
            if result:
                return result

        # Fallback to HuggingFace IndicWhisper
        return self._transcribe_hf(audio_bytes, language)

    def _transcribe_sarvam(self, audio_bytes: bytes, filename: str, language: str) -> str:
        """Transcribe using Sarvam AI Saarika ASR API."""
        try:
            import requests
            lang_code = _SARVAM_LANG.get(language, "hi-IN")
            resp = requests.post(
                f"{SARVAM_API_BASE}/speech-to-text",
                headers={"api-subscription-key": self._sarvam_key},
                files={"file": (filename, audio_bytes, "audio/wav")},
                data={"language_code": lang_code, "model": "saarika:v2.5"},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json().get("transcript", "")
            logger.warning(f"Sarvam ASR returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Sarvam ASR failed: {e}")
        return ""

    def _transcribe_hf(self, audio_bytes: bytes, language: str) -> str:
        """Transcribe via HuggingFace Inference API (fallback)."""
        if not self._hf_key:
            return ""
        try:
            import requests
            model = "ai4bharat/indicwhisper-hindi" if language == "hi" else "openai/whisper-base"
            resp = requests.post(
                f"https://router.huggingface.co/models/{model}",
                headers={"Authorization": f"Bearer {self._hf_key}"},
                data=audio_bytes,
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json().get("text", "")
            logger.warning(f"HF ASR returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"HF ASR fallback failed: {e}")
        return ""

    def transcribe_from_gradio(self, audio_tuple, language: str = "hi") -> str:
        """
        Handle Gradio audio widget output.
        Gradio returns (sample_rate, numpy_array) tuple.
        """
        if audio_tuple is None:
            return ""
        try:
            import numpy as np
            import io
            import wave

            sr, audio_np = audio_tuple
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes((audio_np * 32767).astype(np.int16).tobytes())
            return self.transcribe(buf.getvalue(), language)
        except Exception as e:
            logger.warning(f"Gradio audio processing failed: {e}")
            return ""
