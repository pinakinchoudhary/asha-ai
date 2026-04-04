# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — ASHA Copilot App
# MAGIC Full interactive Gradio application — the main demo notebook.
# MAGIC
# MAGIC **Tabs:**
# MAGIC 1. Voice Copilot — Hindi/English chat with intent routing
# MAGIC 2. Patient CRUD — Add/update patients via natural language
# MAGIC 3. Triage Dashboard — Input vitals → ML triage → doctor/PHC assignment
# MAGIC 4. Scheme Checker — Check welfare scheme eligibility
# MAGIC 5. Protocol Q&A — RAG-powered clinical protocol questions

# COMMAND ----------

# MAGIC %pip install faiss-cpu sentence-transformers PyPDF2 gradio gtts pyyaml requests
# MAGIC %restart_python

# COMMAND ----------

import sys, os
repo_root = os.path.dirname(os.path.dirname(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
))
sys.path.insert(0, f"/Workspace{repo_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize All Components

# COMMAND ----------

from src.models.indic_llm import IndicLLM
from src.models.indic_translate import IndicTranslator
from src.models.indic_asr import IndicASR
from src.models.indic_tts import IndicTTS
from src.triage_engine import TriageEngine
from src.scheme_eligibility import SchemeEngine
from src.rag_pipeline import RAGPipeline
from src.voice_db_agent import VoiceDBAgent
from src.voice_pipeline import VoicePipeline
from config.settings import FAISS_INDEX_PATH

# Load models
print("Initializing LLM (Sarvam API)...")
llm = IndicLLM()
print(f"  LLM API available: {llm.is_loaded}")

translator = IndicTranslator()
asr = IndicASR()
tts = IndicTTS()

# Load engines
print("Loading triage engine...")
triage_engine = TriageEngine(
    llm=llm,
    rules_path=f"/Workspace{repo_root}/config/danger_signs.yaml"
)

print("Loading scheme engine...")
scheme_engine = SchemeEngine(llm=llm)

print("Loading RAG pipeline...")
rag = RAGPipeline(llm=llm)
if os.path.exists(FAISS_INDEX_PATH):
    rag.load_index(FAISS_INDEX_PATH)
    print(f"  FAISS index loaded from {FAISS_INDEX_PATH}")
else:
    print("  No FAISS index found — RAG will use LLM knowledge only")

print("Loading voice DB agent...")
voice_db = VoiceDBAgent(llm=llm, translator=translator, asr=asr, spark=spark)

print("Loading voice pipeline...")
pipeline = VoicePipeline(
    asr=asr, translator=translator, llm=llm, tts=tts,
    triage_engine=triage_engine, voice_db_agent=voice_db,
    scheme_engine=scheme_engine, rag_pipeline=rag,
    spark=spark,
)

print("\nAll components loaded. Starting Gradio app...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Launch Gradio App

# COMMAND ----------

import gradio as gr
import json
import uuid
from datetime import date

# ---- Tab 1: Voice Copilot ----

def copilot_chat(message, language, history):
    """Process a chat message through the full voice pipeline."""
    lang_code = "hi" if language == "Hindi" else "en"
    response = pipeline.process(message, language=lang_code)

    reply = response.text_response_local if lang_code == "hi" else response.text_response_en
    if not reply:
        reply = response.text_response_en or "No response generated."

    # Add structured info if available
    extras = []
    if response.triage_result:
        extras.append(f"\n**Triage**: {response.triage_result.get('risk_level', 'N/A')}")
    if response.scheme_results:
        eligible = [r['scheme_id'] for r in response.scheme_results if r.get('eligible')]
        if eligible:
            extras.append(f"\n**Eligible Schemes**: {', '.join(eligible)}")
    if response.db_changes and response.db_changes.get('success'):
        extras.append(f"\n**DB Update**: {response.db_changes.get('db_action', '')}")

    full_reply = reply + "".join(extras)
    history.append((message, full_reply))
    return history, ""


# ---- Tab 2: Patient CRUD ----

def add_patient_voice(command, language):
    lang_code = "hi" if language == "Hindi" else "en"
    result = voice_db.process_voice_command(command, language=lang_code)
    output = f"**Intent**: {result.get('intent', 'unknown')}\n"
    output += f"**Success**: {result.get('success', False)}\n"
    output += f"**Action**: {result.get('db_action', 'None')}\n\n"
    if result.get('success'):
        output += f"**Response**: {result.get('message_hi', result.get('message_en', ''))}\n\n"
        output += f"**Entities extracted**:\n```json\n{json.dumps(result.get('entities', {}), indent=2, default=str)}\n```"
    else:
        output += f"**Error**: {result.get('message_en', 'Unknown error')}"
    return output

def get_recent_patients():
    rows = spark.sql("""
        SELECT name, age, village, bpl_status, registered_date
        FROM workspace.asha_copilot.patients
        ORDER BY registered_date DESC LIMIT 10
    """).toPandas()
    return rows


# ---- Tab 3: Triage ----

def run_triage(bp_sys, bp_dia, hb, temp, notes):
    visit_data = {
        "bp_systolic": int(bp_sys) if bp_sys else None,
        "bp_diastolic": int(bp_dia) if bp_dia else None,
        "hemoglobin_g_dl": float(hb) if hb else None,
        "temperature_c": float(temp) if temp else None,
        "clinical_notes": notes or "",
        "danger_signs_observed": [],
    }
    result = triage_engine.triage(visit_data, spark=spark)

    color = {"RED": "red", "YELLOW": "orange", "GREEN": "green"}.get(result.risk_level, "gray")
    output = f"## Risk Level: <span style='color:{color}'>{result.risk_level}</span>\n\n"
    output += f"**Method**: {result.method}\n\n"

    output += "### Alerts:\n"
    for a in result.alerts:
        output += f"- **{a.sign_name}** ({a.severity}): {a.action} [Urgency: {a.urgency_hours}h, Confidence: {a.confidence:.2f}]\n"

    if result.assignment and result.assignment.phc_name:
        a = result.assignment
        output += f"\n### Referral Assignment:\n"
        output += f"- **Facility**: {a.phc_name} ({a.phc_type})\n"
        output += f"- **Doctor**: {a.doctor_name} ({a.doctor_specialization})\n"
        output += f"- **Phone**: {a.doctor_phone}\n"

    return output


# ---- Tab 4: Scheme Checker ----

def check_schemes(name):
    if not name:
        return "Please enter a patient name."
    patients = spark.sql(f"""
        SELECT * FROM workspace.asha_copilot.patients
        WHERE lower(name) LIKE '%{name.lower().split()[0]}%'
        LIMIT 1
    """).collect()
    if not patients:
        return f"Patient '{name}' not found."

    p = patients[0].asDict()
    results = scheme_engine.evaluate(p)

    output = f"## Scheme Eligibility for {p['name']}\n"
    output += f"Age: {p['age']} | Village: {p['village']} | BPL: {p['bpl_status']} | Caste: {p['caste_category']}\n\n"

    for r in results:
        status = "Eligible" if r.eligible else "Not Eligible"
        icon = "[Y]" if r.eligible else "[N]"
        output += f"### {icon} {r.scheme_id} — {r.scheme_name}\n"
        output += f"- **Status**: {status}\n"
        if r.amount:
            output += f"- **Amount**: Rs {r.amount:,}\n"
        if r.missing_requirements:
            output += f"- **Missing**: {', '.join(r.missing_requirements)}\n"
        if r.entitlements:
            output += f"- **Entitlements**: {', '.join(r.entitlements[:3])}\n"
        output += f"- {r.notes}\n\n"

    return output


# ---- Tab 5: Protocol Q&A ----

def ask_protocol(question):
    if not question:
        return "Please enter a question."
    result = rag.answer(question)
    output = f"## Answer\n\n{result['answer']}\n\n"
    if result['sources']:
        output += f"**Sources**: {', '.join(result['sources'])}"
    return output


# ---- Build Gradio App ----

with gr.Blocks(
    title="ASHA Copilot",
    theme=gr.themes.Soft(),
) as app:
    gr.Markdown("# ASHA Copilot — AI-Powered Maternal & Child Healthcare Assistant")
    gr.Markdown("Voice-first, multilingual field assistant for India's community health workers")

    with gr.Tab("Voice Copilot"):
        gr.Markdown("### Chat with ASHA Copilot in Hindi or English")
        lang_select = gr.Radio(["Hindi", "English"], value="Hindi", label="Language")
        chatbot = gr.Chatbot(height=400)
        msg_input = gr.Textbox(
            placeholder="Type or paste ASHA's voice input here...",
            label="ASHA Input"
        )
        msg_input.submit(copilot_chat, [msg_input, lang_select, chatbot], [chatbot, msg_input])

        gr.Markdown("**Try these examples:**")
        gr.Examples(
            examples=[
                ["Patient ko kal raat se severe headache aur chakkar aa raha hai. BP check kiya toh 160/100 tha."],
                ["Naya patient register karo — Geeta Devi, umra 26, gaon Sarnath, BPL hai"],
                ["Sunita Devi PMMVY scheme ke liye eligible hai kya?"],
                ["BCG vaccine kab dena chahiye?"],
                ["Meena ka checkup karo — BP 130/85, hemoglobin 8.5, weight 52 kg"],
            ],
            inputs=msg_input,
        )

    with gr.Tab("Patient CRUD"):
        gr.Markdown("### Add/Update Patients via Natural Language")
        with gr.Row():
            crud_lang = gr.Radio(["Hindi", "English"], value="Hindi", label="Language")
        crud_input = gr.Textbox(
            label="Voice Command",
            placeholder="e.g., 'Naya patient register karo — Meena Kumari, umra 22, gaon Sultanpur'",
            lines=2,
        )
        crud_btn = gr.Button("Process Command", variant="primary")
        crud_output = gr.Markdown()
        crud_btn.click(add_patient_voice, [crud_input, crud_lang], crud_output)

        gr.Markdown("### Recent Patients")
        patients_table = gr.Dataframe(value=get_recent_patients, every=10)

    with gr.Tab("Triage"):
        gr.Markdown("### Clinical Triage — Enter Vitals")
        with gr.Row():
            bp_s = gr.Number(label="BP Systolic", value=120)
            bp_d = gr.Number(label="BP Diastolic", value=80)
            hb = gr.Number(label="Hemoglobin (g/dL)", value=11.0)
            temp = gr.Number(label="Temperature (°C)", value=37.0)
        notes = gr.Textbox(label="Clinical Notes / Symptoms", placeholder="Patient complaints...", lines=2)
        triage_btn = gr.Button("Run Triage", variant="primary")
        triage_output = gr.Markdown()
        triage_btn.click(run_triage, [bp_s, bp_d, hb, temp, notes], triage_output)

    with gr.Tab("Scheme Checker"):
        gr.Markdown("### Check Welfare Scheme Eligibility")
        scheme_name = gr.Textbox(label="Patient Name", placeholder="Enter patient name...")
        scheme_btn = gr.Button("Check Eligibility", variant="primary")
        scheme_output = gr.Markdown()
        scheme_btn.click(check_schemes, scheme_name, scheme_output)

    with gr.Tab("Protocol Q&A"):
        gr.Markdown("### Ask Clinical Protocol Questions (RAG-powered)")
        qa_input = gr.Textbox(label="Question", placeholder="What are the danger signs during pregnancy?")
        qa_btn = gr.Button("Ask", variant="primary")
        qa_output = gr.Markdown()
        qa_btn.click(ask_protocol, qa_input, qa_output)

        gr.Examples(
            examples=[
                ["What are the danger signs during pregnancy?"],
                ["What is the catch-up schedule for missed Pentavalent dose?"],
                ["When should a newborn be referred to a health facility?"],
                ["What are the HBNC visit schedules?"],
                ["What is the treatment for severe anemia in pregnancy?"],
            ],
            inputs=qa_input,
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Launch the App

# COMMAND ----------

app.launch()
