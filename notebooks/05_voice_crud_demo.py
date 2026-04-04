# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Voice-Driven CRUD Demo
# MAGIC ASHA adds and updates patient records via voice commands in Hindi/English.
# MAGIC
# MAGIC **Flow**: ASHA speaks (Hindi) → ASR → Sarvam Translate → Sarvam LLM Entity Extraction → Delta Table CRUD → Confirm (Hindi)

# COMMAND ----------

# MAGIC %pip install requests gtts
# MAGIC %restart_python

# COMMAND ----------

import sys, os
repo_root = os.path.dirname(os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
))
sys.path.insert(0, f"/Workspace{repo_root}")

from src.models.indic_llm import IndicLLM
from src.models.indic_translate import IndicTranslator
from src.models.indic_asr import IndicASR
from src.voice_db_agent import VoiceDBAgent

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Initialize Voice DB Agent

# COMMAND ----------

llm = IndicLLM()
translator = IndicTranslator()
asr = IndicASR()

agent = VoiceDBAgent(llm=llm, translator=translator, asr=asr, spark=spark)

print(f"LLM API available: {llm.is_loaded}")
print("Voice DB Agent ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Demo: Register a New Patient (Hindi Voice Input)
# MAGIC
# MAGIC Simulating ASHA speaking in Hindi:
# MAGIC > "Naya patient register karo — Meena Kumari, umra 22 saal, gaon Sultanpur, BPL card hai, Aadhaar hai"

# COMMAND ----------

# Show patients table BEFORE
print("PATIENTS TABLE BEFORE:")
display(
    spark.sql("""
        SELECT name, age, village, bpl_status, registered_date
        FROM workspace.asha_copilot.patients
        ORDER BY registered_date DESC
        LIMIT 5
    """)
)

# COMMAND ----------

hindi_command = "Naya patient register karo — Meena Kumari, umra 22 saal, gaon Sultanpur, BPL card hai, Aadhaar hai"

print(f"ASHA says: \"{hindi_command}\"")
print("-" * 60)

result = agent.process_voice_command(hindi_command, language="hi")

print(f"Intent detected: {result['intent']}")
print(f"Success: {result['success']}")
print(f"DB Action: {result['db_action']}")
print(f"\nResponse (Hindi): {result['message_hi']}")
print(f"Response (English): {result['message_en']}")

# COMMAND ----------

# Show patients table AFTER
print("PATIENTS TABLE AFTER:")
display(
    spark.sql("""
        SELECT name, age, village, bpl_status, aadhaar_registered, registered_date
        FROM workspace.asha_copilot.patients
        WHERE lower(name) LIKE '%meena%'
        ORDER BY registered_date DESC
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Demo: Log a Visit with Vitals (Hindi Voice Input)
# MAGIC
# MAGIC > "Meena Kumari ka checkup — BP 145/95, weight 55 kg, hemoglobin 8.2, sar mein dard hai aur pair sooje hain"

# COMMAND ----------

visit_command = "Meena Kumari ka checkup — BP 145/95, weight 55 kg, hemoglobin 8.2, sar mein dard hai aur pair sooje hain"

print(f"ASHA says: \"{visit_command}\"")
print("-" * 60)

result = agent.process_voice_command(visit_command, language="hi")

print(f"Intent: {result['intent']}")
print(f"Success: {result['success']}")
print(f"DB Action: {result['db_action']}")
print(f"\nExtracted Entities:")
for k, v in result.get('entities', {}).items():
    if v is not None and v != "" and v != []:
        print(f"  {k}: {v}")

print(f"\nResponse (Hindi): {result['message_hi']}")
print(f"Response (English): {result['message_en']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Demo: English Voice Command
# MAGIC
# MAGIC > "Register new patient Lakshmi Yadav, age 28, village Rampur, she has previous C-section history"

# COMMAND ----------

english_command = "Register new patient Lakshmi Yadav, age 28, village Rampur, she has previous C-section history"

print(f"ASHA says: \"{english_command}\"")
print("-" * 60)

result = agent.process_voice_command(english_command, language="en")

print(f"Intent: {result['intent']}")
print(f"Success: {result['success']}")
print(f"Response: {result['message_en']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Demo: Check Patient Status

# COMMAND ----------

display(
    spark.sql("""
        SELECT p.name, p.age, p.village, p.bpl_status,
               v.visit_date, v.bp_systolic, v.bp_diastolic, v.hemoglobin_g_dl,
               v.clinical_notes
        FROM workspace.asha_copilot.patients p
        LEFT JOIN workspace.asha_copilot.visits v ON p.patient_id = v.patient_id
        WHERE lower(p.name) LIKE '%meena%'
        ORDER BY v.visit_date DESC
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The Voice DB Agent successfully:
# MAGIC 1. Parsed Hindi natural language commands
# MAGIC 2. Classified intent (add_patient, log_visit) via Sarvam LLM
# MAGIC 3. Extracted structured entities from unstructured speech
# MAGIC 4. Validated data before writing to Delta tables
# MAGIC 5. Confirmed actions back in Hindi
