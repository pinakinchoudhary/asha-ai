"""
Indic Text-to-Speech.

Primary: AI4Bharat TTS API (natural Indic voices)
Fallback: gTTS (Google TTS — decent Hindi support)
"""

import logging
import os

logger = logging.getLogger(__name__)


class IndicTTS:
    """Text-to-speech for Hindi and English."""

    def __init__(self):
        self._api_key = os.environ.get("SARVAM_API_KEY", "")

    def synthesize(self, text: str, language: str = "hi") -> bytes:
        """
        Convert text to speech audio bytes (MP3).

        Args:
            text: Text to speak.
            language: Language code ("hi" or "en").

        Returns:
            Audio bytes (MP3 format).
        """
        if not text or not text.strip():
            return b""

        # Try AI4Bharat / Sarvam TTS API first
        audio = self._synthesize_api(text, language)
        if audio:
            return audio

        # Fallback to gTTS
        return self._synthesize_gtts(text, language)

    def _synthesize_api(self, text: str, language: str) -> bytes:
        """Synthesize using Sarvam AI TTS API."""
        if not self._api_key:
            return b""
        try:
            import requests
            resp = requests.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": text,
                    "language_code": language,
                    "model": "sarvam-tts-v1",
                },
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.content
            logger.warning(f"Sarvam TTS API returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"Sarvam TTS API failed: {e}")
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
