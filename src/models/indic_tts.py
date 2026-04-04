"""
Indic Text-to-Speech.

Primary: Sarvam AI Bulbul TTS API (natural Indic voices)
Fallback: gTTS (Google TTS — decent Hindi support)
"""

import base64
import logging
import os

logger = logging.getLogger(__name__)

SARVAM_API_BASE = "https://api.sarvam.ai"

_SARVAM_LANG = {
    "hi": "hi-IN",
    "en": "en-IN",
    "ta": "ta-IN",
    "bn": "bn-IN",
}


class IndicTTS:
    """Text-to-speech for Hindi and English."""

    def __init__(self):
        self._api_key = os.environ.get("SARVAM_API_KEY", "")

    def synthesize(self, text: str, language: str = "hi") -> bytes:
        """
        Convert text to speech audio bytes (WAV/MP3).

        Args:
            text: Text to speak.
            language: Language code ("hi" or "en").

        Returns:
            Audio bytes.
        """
        if not text or not text.strip():
            return b""

        # Try Sarvam Bulbul TTS first
        if self._api_key:
            audio = self._synthesize_sarvam(text, language)
            if audio:
                return audio

        # Fallback to gTTS
        return self._synthesize_gtts(text, language)

    def _synthesize_sarvam(self, text: str, language: str) -> bytes:
        """Synthesize using Sarvam AI Bulbul TTS API."""
        try:
            import requests
            lang_code = _SARVAM_LANG.get(language, "hi-IN")
            resp = requests.post(
                f"{SARVAM_API_BASE}/text-to-speech",
                headers={
                    "api-subscription-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": [text[:500]],  # Sarvam max ~500 chars per request
                    "target_language_code": lang_code,
                    "speaker": "meera",
                    "model": "bulbul:v2",
                    "enable_preprocessing": True,
                },
                timeout=20,
            )
            if resp.status_code == 200:
                data = resp.json()
                audios = data.get("audios", [])
                if audios:
                    # Sarvam returns base64-encoded WAV
                    return base64.b64decode(audios[0])
            logger.warning(f"Sarvam TTS returned {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Sarvam TTS failed: {e}")
        return b""

    def _synthesize_gtts(self, text: str, language: str) -> bytes:
        """Synthesize using Google TTS (fallback)."""
        try:
            from gtts import gTTS
            import io

            lang_map = {"hi": "hi", "en": "en"}
            tts = gTTS(text=text, lang=lang_map.get(language, "hi"), slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            logger.warning(f"gTTS fallback failed: {e}")
            return b""
