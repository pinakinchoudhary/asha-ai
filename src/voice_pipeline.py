"""
End-to-end Voice Pipeline.

ASR → Translate → Intent Detection → Route to Engine → Translate Back → TTS
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VoiceResponse:
    text_response_en: str = ""
    text_response_local: str = ""
    audio_bytes: bytes = b""
    intent: str = ""
    triage_result: dict = field(default_factory=dict)
    db_changes: dict = field(default_factory=dict)
    scheme_results: list = field(default_factory=list)
    rag_answer: str = ""


class VoicePipeline:
    """
    Orchestrates the full voice-first interaction loop.

    Components:
        asr: IndicASR — speech to text
        translator: IndicTranslator — Hindi ↔ English
        llm: IndicLLM — intent classification, entity extraction, reasoning
        tts: IndicTTS — text to speech
        triage_engine: TriageEngine — clinical danger sign detection
        voice_db_agent: VoiceDBAgent — CRUD via voice
        scheme_engine: SchemeEngine — welfare scheme eligibility
        rag_pipeline: RAGPipeline — protocol Q&A
    """

    def __init__(self, asr, translator, llm, tts,
                 triage_engine=None, voice_db_agent=None,
                 scheme_engine=None, rag_pipeline=None,
                 spark=None):
        self.asr = asr
        self.translator = translator
        self.llm = llm
        self.tts = tts
        self.triage_engine = triage_engine
        self.voice_db_agent = voice_db_agent
        self.scheme_engine = scheme_engine
        self.rag_pipeline = rag_pipeline
        self.spark = spark

    def process(self, audio_or_text, language: str = "hi",
                patient_context: dict = None) -> VoiceResponse:
        """
        Process a complete voice interaction.

        Args:
            audio_or_text: Audio bytes/path or text string.
            language: Source language code ("hi" or "en").
            patient_context: Optional dict with current patient info for context.

        Returns:
            VoiceResponse with text, audio, and structured results.
        """
        response = VoiceResponse()

        # Step 1: ASR — transcribe audio to text
        if isinstance(audio_or_text, str):
            text_local = audio_or_text
        else:
            text_local = self.asr.transcribe(audio_or_text, language)

        if not text_local.strip():
            response.text_response_en = "No speech detected."
            response.text_response_local = "Koi awaaz nahi mili."
            return response

        # Step 2: Translate to English
        if language != "en":
            text_en = self.translator.to_english(text_local, src_lang=language)
        else:
            text_en = text_local

        # Step 3: Classify intent
        intent = self.llm.classify_intent(text_en)
        response.intent = intent

        # Step 4: Route to appropriate engine
        if intent in ("add_patient", "update_record", "log_visit"):
            response = self._handle_crud(text_local, text_en, language, intent, response)
        elif intent == "query_patient":
            response = self._handle_query_patient(text_local, language, response)
        elif intent == "symptom_report":
            response = self._handle_symptom(text_local, text_en, language, patient_context, response)
        elif intent == "scheme_query":
            response = self._handle_scheme(text_en, language, patient_context, response)
        elif intent == "immunization_check":
            response = self._handle_immunization(text_en, language, patient_context, response)
        elif intent == "protocol_question":
            response = self._handle_protocol(text_en, language, response)
        else:
            response.text_response_en = "I didn't understand that. Could you rephrase?"
            response.text_response_local = "Mujhe samajh nahi aaya. Kya aap dobara bata sakti hain?"

        # Step 5: Translate response back to local language
        if language != "en" and response.text_response_en:
            if not response.text_response_local:
                response.text_response_local = self.translator.from_english(
                    response.text_response_en, tgt_lang=language
                )

        # Step 6: TTS — generate audio
        speak_text = response.text_response_local or response.text_response_en
        if speak_text:
            response.audio_bytes = self.tts.synthesize(speak_text, language)

        return response

    # ------------------------------------------------------------------
    # Intent handlers
    # ------------------------------------------------------------------

    def _handle_query_patient(self, text_local, language, response):
        """Handle query_patient intent — look up patients in DB."""
        if self.voice_db_agent:
            result = self.voice_db_agent.process_voice_command(text_local, language)
            response.db_changes = result
            response.text_response_en = result.get("message_en", "")
            response.text_response_local = result.get("message_hi", "")
        return response

    def _handle_crud(self, text_local, text_en, language, intent, response):
        """Handle add/update/log_visit intents."""
        if self.voice_db_agent:
            result = self.voice_db_agent.process_voice_command(text_local, language)
            response.db_changes = result
            response.text_response_en = result.get("message_en", "")
            response.text_response_local = result.get("message_hi", "")

            # If it was a visit log, also run triage
            if intent == "log_visit" and result.get("success") and self.triage_engine:
                visit_data = result.get("entities", {})
                triage = self.triage_engine.triage(visit_data, spark=self.spark)
                response.triage_result = triage.to_dict()

                # Append triage info to response
                if triage.risk_level in ("RED", "YELLOW"):
                    triage_msg = self._format_triage_message(triage)
                    response.text_response_en += f"\n\n{triage_msg['en']}"
                    response.text_response_local += f"\n\n{triage_msg['hi']}"
        else:
            response.text_response_en = "Database agent not available."
            response.text_response_local = "Database agent uplabdh nahi hai."
        return response

    def _handle_symptom(self, text_local, text_en, language, patient_context, response):
        """Handle symptom reports — run triage."""
        # Extract entities for visit data
        entities = self.llm.extract_entities(text_en, "symptom_report")
        visit_data = {
            "bp_systolic": entities.get("bp_systolic"),
            "bp_diastolic": entities.get("bp_diastolic"),
            "hemoglobin_g_dl": entities.get("hemoglobin"),
            "temperature_c": entities.get("temperature"),
            "clinical_notes": text_local,
            "danger_signs_observed": entities.get("symptoms", []),
        }
        if patient_context:
            visit_data["patient_id"] = patient_context.get("patient_id", "")
            visit_data["village"] = patient_context.get("village", "")
            visit_data["nearest_phc_id"] = patient_context.get("nearest_phc_id", "")

        if self.triage_engine:
            triage = self.triage_engine.triage(visit_data, spark=self.spark)
            response.triage_result = triage.to_dict()
            msg = self._format_triage_message(triage)
            response.text_response_en = msg["en"]
            response.text_response_local = msg["hi"]
        else:
            response.text_response_en = "Triage engine not available."
        return response

    def _handle_scheme(self, text_en, language, patient_context, response):
        """Handle scheme eligibility queries."""
        if self.scheme_engine and patient_context:
            results = self.scheme_engine.evaluate(patient_context)
            response.scheme_results = [r.to_dict() for r in results]
            eligible = [r for r in results if r.eligible]
            if eligible:
                schemes_str = ", ".join([f"{r.scheme_id} (Rs {r.amount})" for r in eligible])
                response.text_response_en = f"Patient is eligible for: {schemes_str}"
                response.text_response_local = f"Patient {schemes_str} ke liye eligible hai."
            else:
                response.text_response_en = "Patient is not currently eligible for any schemes."
                response.text_response_local = "Patient abhi kisi scheme ke liye eligible nahi hai."
        elif self.scheme_engine:
            # Try extracting patient info from text
            entities = self.llm.extract_entities(text_en, "scheme_query")
            if entities:
                results = self.scheme_engine.evaluate(entities)
                response.scheme_results = [r.to_dict() for r in results]
                eligible = [r for r in results if r.eligible]
                if eligible:
                    schemes_str = ", ".join([r.scheme_id for r in eligible])
                    response.text_response_en = f"Based on provided info, eligible for: {schemes_str}"
                    response.text_response_local = f"Di gayi jaankari ke hisab se eligible: {schemes_str}"
                else:
                    response.text_response_en = "Not eligible based on provided information."
                    response.text_response_local = "Di gayi jaankari se eligible nahi lag raha."
        return response

    def _handle_immunization(self, text_en, language, patient_context, response):
        """Handle immunization schedule queries."""
        if self.spark and patient_context:
            mother_id = patient_context.get("patient_id", "")
            try:
                overdue = self.spark.sql(f"""
                    SELECT vaccine_code, due_date, child_name
                    FROM workspace.asha_copilot.immunizations
                    WHERE mother_id = '{mother_id}' AND missed_flag = true
                    ORDER BY due_date
                """).collect()
                if overdue:
                    vaccines = ", ".join([r["vaccine_code"] for r in overdue])
                    response.text_response_en = f"Overdue vaccines: {vaccines}. Please schedule immediately."
                    response.text_response_local = f"Overdue vaccines: {vaccines}. Jaldi schedule karein."
                else:
                    response.text_response_en = "All vaccinations are up to date."
                    response.text_response_local = "Sab teekakaran samay par hua hai."
            except Exception as e:
                response.text_response_en = f"Could not check immunization records: {e}"
        else:
            response.text_response_en = "Immunization records not available in current context."
            response.text_response_local = "Teekakaran ka record abhi uplabdh nahi hai."
        return response

    def _handle_protocol(self, text_en, language, response):
        """Handle clinical protocol questions via RAG."""
        if self.rag_pipeline:
            result = self.rag_pipeline.answer(text_en)
            response.rag_answer = result.get("answer", "")
            response.text_response_en = result.get("answer", "")
            sources = result.get("sources", [])
            if sources:
                response.text_response_en += f"\n\nSources: {', '.join(sources)}"
        else:
            response.text_response_en = (
                "Protocol database not available. Please consult your supervisor "
                "or refer to the NHM guideline booklet."
            )
            response.text_response_local = (
                "Protocol database uplabdh nahi hai. Apne supervisor se poochhein "
                "ya NHM guideline booklet dekhein."
            )
        return response

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_triage_message(self, triage) -> dict:
        """Format triage result into Hindi + English messages."""
        level = triage.risk_level
        level_emoji = {"RED": "[URGENT]", "YELLOW": "[CAUTION]", "GREEN": "[OK]"}.get(level, "")
        level_hi = {"RED": "[ATYANT ZARURI]", "YELLOW": "[SAVDHAN]", "GREEN": "[THEEK]"}.get(level, "")

        alerts_en = "; ".join([f"{a.sign_name}: {a.action}" for a in triage.alerts[:3]])
        en = f"{level_emoji} Risk Level: {level}. {alerts_en}"
        hi = f"{level_hi} Risk Level: {level}. {alerts_en}"

        if triage.assignment and triage.assignment.phc_name:
            a = triage.assignment
            en += f"\nRefer to: {a.phc_name} ({a.phc_type}). Doctor: {a.doctor_name} ({a.doctor_specialization}). Phone: {a.doctor_phone}"
            hi += f"\nBhejein: {a.phc_name} ({a.phc_type}). Doctor: {a.doctor_name} ({a.doctor_specialization}). Phone: {a.doctor_phone}"

        return {"en": en, "hi": hi}
