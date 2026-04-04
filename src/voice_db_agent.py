"""
Voice-Driven Database CRUD Agent.

ASHA speaks in Hindi/English → system parses intent → extracts entities → updates Delta tables.
"""

import json
import logging
import uuid
from datetime import date, timedelta

logger = logging.getLogger(__name__)


class VoiceDBAgent:
    """
    Processes voice commands from ASHA workers to add/update patient records.

    Flow: Speech → ASR → Translate → Intent + Entity extraction → Delta CRUD → Confirm
    """

    def __init__(self, llm, translator, asr=None, spark=None):
        self.llm = llm
        self.translator = translator
        self.asr = asr
        self.spark = spark

    def process_voice_command(self, audio_or_text, language: str = "hi") -> dict:
        """
        Process a voice command end-to-end.

        Args:
            audio_or_text: Audio bytes/path or text string.
            language: Source language ("hi" or "en").

        Returns:
            dict with keys: success, intent, entities, db_action, message_hi, message_en
        """
        # Step 1: ASR — audio to text
        if self.asr and not isinstance(audio_or_text, str):
            text = self.asr.transcribe(audio_or_text, language)
        else:
            text = audio_or_text if isinstance(audio_or_text, str) else ""

        if not text.strip():
            return self._error("No speech detected / Koi awaaz nahi mili")

        # Step 2: Translate to English if needed
        if language != "en" and self.translator:
            text_en = self.translator.to_english(text, src_lang=language)
        else:
            text_en = text

        # Step 3: Intent classification
        intent = self.llm.classify_intent(text_en)

        # Step 4: Entity extraction
        entities = self.llm.extract_entities(text_en, intent)

        # Step 5: Validate entities
        valid, errors = self._validate_entities(entities, intent)
        if not valid:
            msg = f"Missing or invalid information: {', '.join(errors)}"
            msg_hi = f"Kuch jaankari galat ya adhuri hai: {', '.join(errors)}"
            return {
                "success": False, "intent": intent, "entities": entities,
                "db_action": None, "message_en": msg, "message_hi": msg_hi,
                "original_text": text, "translated_text": text_en,
            }

        # Step 6: Execute DB operation
        if intent == "add_patient":
            return self._add_patient(entities, text, text_en)
        elif intent in ("log_visit", "update_record", "symptom_report"):
            return self._log_visit(entities, text, text_en)
        else:
            return {
                "success": True, "intent": intent, "entities": entities,
                "db_action": None,
                "message_en": f"Intent '{intent}' does not require a DB update.",
                "message_hi": f"'{intent}' ke liye database update zaroori nahi hai.",
                "original_text": text, "translated_text": text_en,
            }

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    def _add_patient(self, entities: dict, original: str, translated: str) -> dict:
        """Register a new patient in the patients Delta table."""
        name = entities.get("name", "Unknown")
        age = entities.get("age")
        village = entities.get("village", "Unknown")
        patient_id = str(uuid.uuid4())
        today = date.today()

        # Compute LMP/EDD if provided
        lmp_str = entities.get("lmp_date")
        if lmp_str:
            try:
                lmp = date.fromisoformat(lmp_str)
            except (ValueError, TypeError):
                lmp = today - timedelta(weeks=12)  # default 12 weeks ago
        else:
            lmp = today - timedelta(weeks=12)
        edd = lmp + timedelta(days=280)

        row = {
            "patient_id": patient_id,
            "name": name,
            "age": age or 25,
            "husband_name": entities.get("husband_name", ""),
            "state": self._infer_state(village),
            "district": self._infer_district(village),
            "village": village,
            "nearest_phc_id": self._infer_phc(village),
            "caste_category": entities.get("caste_category", "OBC"),
            "bpl_status": entities.get("bpl_status", False),
            "blood_group": "Unknown",
            "gravida": entities.get("gravida", 1),
            "para": entities.get("para", 0),
            "abortions": 0,
            "lmp": str(lmp),
            "edd": str(edd),
            "risk_flags": [],
            "aadhaar_registered": entities.get("aadhaar", False),
            "bank_account": entities.get("bank_account", False),
            "phone": "",
            "registered_date": str(today),
        }

        if self.spark:
            try:
                df = self.spark.createDataFrame([row])
                df.write.format("delta").mode("append").saveAsTable(
                    "hive_metastore.asha_copilot.patients"
                )
                msg_en = f"Patient {name} (age {row['age']}) registered successfully. ID: {patient_id[:8]}..."
                msg_hi = f"{name} (umra {row['age']}) ka registration ho gaya. ID: {patient_id[:8]}..."
                return {
                    "success": True, "intent": "add_patient", "entities": row,
                    "db_action": "INSERT into patients", "message_en": msg_en,
                    "message_hi": msg_hi, "original_text": original,
                    "translated_text": translated,
                }
            except Exception as e:
                return self._error(f"Database write failed: {e}")
        else:
            return {
                "success": True, "intent": "add_patient", "entities": row,
                "db_action": "INSERT (dry run — no Spark session)",
                "message_en": f"[Dry run] Would register {name}",
                "message_hi": f"[Dry run] {name} register hota",
                "original_text": original, "translated_text": translated,
            }

    def _log_visit(self, entities: dict, original: str, translated: str) -> dict:
        """Log an ANC visit with vitals to the visits Delta table."""
        name = entities.get("name", "")
        village = entities.get("village", "")

        # Find patient
        patient_id = None
        if self.spark and name:
            patient_id = self._find_patient(name, village)

        if not patient_id and self.spark:
            return self._error(
                f"Patient '{name}' not found. Register first. / "
                f"'{name}' ka record nahi mila. Pehle register karein."
            )

        visit_id = str(uuid.uuid4())
        visit_date = str(date.today())

        row = {
            "visit_id": visit_id,
            "patient_id": patient_id or "unknown",
            "visit_number": 1,
            "visit_date": visit_date,
            "gestational_age_weeks": None,
            "weight_kg": entities.get("weight_kg"),
            "fundal_height_cm": None,
            "bp_systolic": entities.get("bp_systolic"),
            "bp_diastolic": entities.get("bp_diastolic"),
            "hemoglobin_g_dl": entities.get("hemoglobin"),
            "temperature_c": entities.get("temperature"),
            "danger_signs_observed": entities.get("symptoms", []),
            "clinical_notes": original,  # Store original Hindi notes
            "referral_flag": False,
        }

        if self.spark:
            try:
                df = self.spark.createDataFrame([row])
                df.write.format("delta").mode("append").saveAsTable(
                    "hive_metastore.asha_copilot.visits"
                )
                vitals_summary = []
                if row["bp_systolic"]:
                    vitals_summary.append(f"BP {row['bp_systolic']}/{row['bp_diastolic']}")
                if row["hemoglobin_g_dl"]:
                    vitals_summary.append(f"Hb {row['hemoglobin_g_dl']}")
                if row["weight_kg"]:
                    vitals_summary.append(f"Weight {row['weight_kg']}kg")
                vitals_str = ", ".join(vitals_summary)

                msg_en = f"Visit recorded for {name}. Vitals: {vitals_str}"
                msg_hi = f"{name} ki visit record ho gayi. Vitals: {vitals_str}"
                return {
                    "success": True, "intent": "log_visit", "entities": row,
                    "db_action": "INSERT into visits", "message_en": msg_en,
                    "message_hi": msg_hi, "original_text": original,
                    "translated_text": translated,
                }
            except Exception as e:
                return self._error(f"Database write failed: {e}")
        else:
            return {
                "success": True, "intent": "log_visit", "entities": row,
                "db_action": "INSERT (dry run)",
                "message_en": f"[Dry run] Would log visit for {name}",
                "message_hi": f"[Dry run] {name} ki visit record hoti",
                "original_text": original, "translated_text": translated,
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_patient(self, name: str, village: str = "") -> str:
        """Find patient ID by name (fuzzy) and optional village."""
        try:
            query = f"""
                SELECT patient_id, name, village
                FROM hive_metastore.asha_copilot.patients
                WHERE lower(name) LIKE '%{name.lower().split()[0]}%'
            """
            if village:
                query += f" AND lower(village) LIKE '%{village.lower()}%'"
            query += " LIMIT 1"
            rows = self.spark.sql(query).collect()
            if rows:
                return rows[0]["patient_id"]
        except Exception as e:
            logger.warning(f"Patient lookup failed: {e}")
        return None

    def _validate_entities(self, entities: dict, intent: str) -> tuple:
        """Validate extracted entities before DB write."""
        errors = []
        if intent == "add_patient":
            if not entities.get("name"):
                errors.append("Patient name is required / Naam zaroori hai")
        elif intent in ("log_visit", "update_record", "symptom_report"):
            if not entities.get("name"):
                errors.append("Patient name is required / Naam zaroori hai")
            # At least one vital or symptom should be present
            has_vital = any(entities.get(k) for k in [
                "bp_systolic", "weight_kg", "hemoglobin", "temperature", "symptoms"
            ])
            if not has_vital:
                errors.append("At least one vital/symptom needed / Kam se kam ek vital zaroori hai")

        # Sanity checks on values
        bp = entities.get("bp_systolic")
        if bp is not None and (bp < 50 or bp > 250):
            errors.append(f"BP systolic {bp} seems invalid")
        hb = entities.get("hemoglobin")
        if hb is not None and (hb < 2.0 or hb > 20.0):
            errors.append(f"Hemoglobin {hb} seems invalid")
        age = entities.get("age")
        if age is not None and (age < 15 or age > 49):
            errors.append(f"Age {age} outside expected range 15-49")

        return (len(errors) == 0, errors)

    def _infer_state(self, village: str) -> str:
        from src.synthetic_data import STATES_DISTRICTS_VILLAGES
        for state, districts in STATES_DISTRICTS_VILLAGES.items():
            for district, villages in districts.items():
                if village in villages:
                    return state
        return "Unknown"

    def _infer_district(self, village: str) -> str:
        from src.synthetic_data import STATES_DISTRICTS_VILLAGES
        for state, districts in STATES_DISTRICTS_VILLAGES.items():
            for district, villages in districts.items():
                if village in villages:
                    return district
        return "Unknown"

    def _infer_phc(self, village: str) -> str:
        from src.synthetic_data import VILLAGE_PHC_MAP
        return VILLAGE_PHC_MAP.get(village, "PHC-001")

    def _error(self, msg: str) -> dict:
        return {
            "success": False, "intent": None, "entities": {},
            "db_action": None, "message_en": msg, "message_hi": msg,
            "original_text": "", "translated_text": "",
        }
