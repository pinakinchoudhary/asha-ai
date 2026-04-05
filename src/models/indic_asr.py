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
        Convert Gradio microphone output → WAV bytes → transcribe.

        Gradio returns (sample_rate: int, audio: np.ndarray).
        The array dtype can be int16, int32, float32, or float64.
        Shape can be (N,) mono or (N, channels) stereo.
        """
        if audio_tuple is None:
            return ""
        try:
            import numpy as np
            import io
            import wave

            sr, audio_np = audio_tuple

            if audio_np is None or audio_np.size == 0:
                logger.warning("ASR: received empty audio array from Gradio.")
                return ""

            # Stereo → mono
            if audio_np.ndim == 2:
                audio_np = audio_np.mean(axis=1)

            # Normalise to int16 PCM regardless of incoming dtype
            if audio_np.dtype == np.int16:
                audio_int16 = audio_np
            elif audio_np.dtype == np.int32:
                audio_int16 = (audio_np >> 16).astype(np.int16)
            else:
                # float32 / float64: values are in [-1.0, 1.0]
                audio_float = audio_np.astype(np.float32)
                audio_float = np.clip(audio_float, -1.0, 1.0)
                audio_int16 = (audio_float * 32767).astype(np.int16)

            # Minimum length check (~0.1 s)
            if len(audio_int16) < sr * 0.1:
                logger.warning("ASR: audio too short to transcribe.")
                return ""

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)   # 16-bit = 2 bytes
                wf.setframerate(sr)
                wf.writeframes(audio_int16.tobytes())
            wav_bytes = buf.getvalue()
            logger.info(f"ASR: sending {len(wav_bytes)} bytes, {len(audio_int16)/sr:.1f}s, sr={sr}")
            return self.transcribe(wav_bytes, language)
        except Exception as e:
            logger.warning(f"Gradio audio processing failed: {e}")
            return ""
