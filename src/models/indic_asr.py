"""
Indic Automatic Speech Recognition (ASR).

Primary: AI4Bharat IndicWhisper via Hugging Face Inference API
Fallback: Accept text input directly (for demo without mic)
"""

import base64
import logging
import os

logger = logging.getLogger(__name__)


class IndicASR:
    """Speech-to-text for Hindi and English using IndicWhisper."""

    def __init__(self):
        self._api_key = os.environ.get("HF_TOKEN", "")
        self._model_id = "ai4bharat/indicwhisper-hindi"

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
        elif isinstance(audio_input, bytes):
            audio_bytes = audio_input
        else:
            logger.warning("ASR: Invalid input type. Returning empty string.")
            return ""

        # Try API-based transcription
        return self._transcribe_api(audio_bytes, language)

    def _transcribe_api(self, audio_bytes: bytes, language: str) -> str:
        """Transcribe via Hugging Face Inference API."""
        try:
            import requests

            model = self._model_id if language == "hi" else "openai/whisper-base"
            resp = requests.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers={"Authorization": f"Bearer {self._api_key}"},
                data=audio_bytes,
                timeout=30,
            )
            if resp.status_code == 200:
                result = resp.json()
                return result.get("text", "")
            logger.warning(f"ASR API returned {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.warning(f"ASR API failed: {e}")
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
            # Convert numpy array to WAV bytes
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
