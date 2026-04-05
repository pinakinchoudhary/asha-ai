"""
ASHA Copilot — Databricks App entrypoint.

Run via:  python app/main.py
Deployed via: app.yaml  →  command: ["python", "app/main.py"]

The platform injects GRADIO_SERVER_NAME / GRADIO_SERVER_PORT / GRADIO_ROOT_PATH
so demo.launch() is called with NO arguments.
"""

from __future__ import annotations

import base64
import logging
import os
import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
# app/main.py lives one level below repo root
_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_ROOT), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Monkey-patch gradio_client 1.3.0 schema bug ─────────────────────────────
import gradio_client.utils as _gc_utils  # noqa: E402

_orig_inner = _gc_utils._json_schema_to_python_type
_orig_get_type = _gc_utils.get_type


def _safe_inner(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_inner(schema, defs)


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


_gc_utils._json_schema_to_python_type = _safe_inner
_gc_utils.get_type = _safe_get_type
# ── End monkey-patch ─────────────────────────────────────────────────────────

import gradio as gr  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Browser-side audio recorder (pure JS — no gr.Audio involved)
# Uses Web Audio API → 16 kHz mono 16-bit PCM → WAV → base64.
# Stores result in window.__asha_wav_b64 for the Send Voice button.
# ---------------------------------------------------------------------------
_RECORDER_HTML = """
<div style="display:flex;align-items:center;gap:12px;padding:8px 0;">
  <button id="asha-rec-btn"
    style="padding:10px 24px;font-size:15px;border-radius:8px;border:none;
           background:#2563eb;color:#fff;cursor:pointer;font-weight:600;"
    onclick="window.__ashaToggleRec()">
    Start Recording
  </button>
  <span id="asha-rec-status" style="font-size:14px;color:#666;">
    Ready to record
  </span>
</div>
<script>
(function(){
  var ctx, stream, src, proc, chunks, recording=false, sr=16000;
  window.__asha_wav_b64 = '';

  window.__ashaToggleRec = async function(){
    var btn=document.getElementById('asha-rec-btn'),
        st=document.getElementById('asha-rec-status');
    if(!recording){
      try{
        ctx = new (window.AudioContext||window.webkitAudioContext)({sampleRate:sr});
        sr = ctx.sampleRate;                       // actual rate browser gave us
        stream = await navigator.mediaDevices.getUserMedia({audio:true});
        src = ctx.createMediaStreamSource(stream);
        proc = ctx.createScriptProcessor(4096,1,1);
        chunks = [];
        proc.onaudioprocess = function(e){
          if(recording) chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        };
        src.connect(proc); proc.connect(ctx.destination);
        recording = true;
        btn.textContent = 'Stop Recording';
        btn.style.background = '#dc2626';
        st.textContent = 'Recording\u2026';
        st.style.color = '#dc2626';
      }catch(err){
        st.textContent = 'Mic error: '+err.message;
        st.style.color = '#dc2626';
      }
    } else {
      recording = false;
      proc.disconnect(); src.disconnect();
      stream.getTracks().forEach(function(t){t.stop();});
      // concat PCM
      var len=0; chunks.forEach(function(c){len+=c.length;});
      var pcm=new Float32Array(len), off=0;
      chunks.forEach(function(c){pcm.set(c,off);off+=c.length;});
      // float32 → int16
      var i16=new Int16Array(pcm.length);
      for(var i=0;i<pcm.length;i++){
        var s=pcm[i]<-1?-1:pcm[i]>1?1:pcm[i];
        i16[i]=s<0?s*32768:s*32767;
      }
      // build WAV
      var nb=44+i16.length*2, buf=new ArrayBuffer(nb), v=new DataView(buf);
      function ws(o,s){for(var j=0;j<s.length;j++)v.setUint8(o+j,s.charCodeAt(j));}
      ws(0,'RIFF'); v.setUint32(4,nb-8,true); ws(8,'WAVE');
      ws(12,'fmt '); v.setUint32(16,16,true); v.setUint16(20,1,true);
      v.setUint16(22,1,true); v.setUint32(24,sr,true);
      v.setUint32(28,sr*2,true); v.setUint16(32,2,true); v.setUint16(34,16,true);
      ws(36,'data'); v.setUint32(40,i16.length*2,true);
      for(var i=0;i<i16.length;i++) v.setInt16(44+i*2,i16[i],true);
      // to base64
      var bytes=new Uint8Array(buf), bin='';
      for(var i=0;i<bytes.length;i++) bin+=String.fromCharCode(bytes[i]);
      window.__asha_wav_b64 = btoa(bin);
      var durSec=(pcm.length/sr).toFixed(1);
      var sizeKb=(nb/1024).toFixed(0);
      ctx.close();
      btn.textContent='Start Recording';
      btn.style.background='#2563eb';
      st.textContent=durSec+'s recorded ('+sizeKb+' KB) \u2014 click Send Voice';
      st.style.color='#16a34a';
    }
  };
})();
</script>
"""

_SECRETS_SCOPE = "asha-ai"
_SECRET_KEYS = {
    "SARVAM_API_KEY": "sarvam-api-key",
    "HF_TOKEN": "hf-token",
    "GROQ_API_KEY": "groq-api-key",
}


# ── Secret loading ────────────────────────────────────────────────────────────

def _load_secrets() -> None:
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        for env_var, secret_key in _SECRET_KEYS.items():
            if os.environ.get(env_var, "").strip():
                continue
            try:
                val = w.secrets.get_secret(scope=_SECRETS_SCOPE, key=secret_key)
                if val and val.value:
                    try:
                        decoded = base64.b64decode(val.value).decode("utf-8")
                    except Exception:
                        decoded = val.value
                    os.environ[env_var] = decoded
                    logger.info("Loaded %s from secret scope %s", env_var, _SECRETS_SCOPE)
            except Exception as exc:
                logger.warning("Could not load %s from secrets: %s", env_var, exc)
    except Exception as exc:
        logger.warning("Secret loading skipped (not in Databricks App?): %s", exc)


# ── Spark via databricks-connect ──────────────────────────────────────────────

def _get_spark():
    try:
        from databricks.connect import DatabricksSession
        spark = DatabricksSession.builder.serverless(True).getOrCreate()
        logger.info("Spark session established via databricks-connect (serverless)")
        return spark
    except Exception as exc:
        logger.warning("databricks-connect unavailable — Spark features disabled: %s", exc)
        return None


# ── Build Gradio app ──────────────────────────────────────────────────────────

def build_app(spark):
    import json
    from datetime import date

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

    logger.info("Initializing models...")
    llm = IndicLLM()
    translator = IndicTranslator()
    asr = IndicASR()
    tts = IndicTTS()

    rules_path = str(_ROOT / "config" / "danger_signs.yaml")
    triage_engine = TriageEngine(llm=llm, rules_path=rules_path)
    scheme_engine = SchemeEngine(llm=llm)

    rag = RAGPipeline(llm=llm)
    if os.path.exists(FAISS_INDEX_PATH):
        rag.load_index(FAISS_INDEX_PATH)

    voice_db = VoiceDBAgent(llm=llm, translator=translator, asr=asr, spark=spark)
    pipeline = VoicePipeline(
        asr=asr, translator=translator, llm=llm, tts=tts,
        triage_engine=triage_engine, voice_db_agent=voice_db,
        scheme_engine=scheme_engine, rag_pipeline=rag,
        spark=spark,
    )
    logger.info("All components loaded.")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _wav_bytes_to_numpy(audio_bytes: bytes):
        """Convert WAV bytes → (sample_rate, np.ndarray) for gr.Audio output."""
        if not audio_bytes:
            return None
        try:
            import io, wave
            import numpy as np
            buf = io.BytesIO(audio_bytes)
            with wave.open(buf, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            return (sr, arr)
        except Exception:
            return None

    # ── Handlers ───────────────────────────────────────────────────────────

    def _process_text(text, lang_code, history):
        """Common pipeline call for both text and voice paths."""
        response = pipeline.process(text, language=lang_code)
        reply = response.text_response_local if lang_code == "hi" else response.text_response_en
        if not reply:
            reply = response.text_response_en or "No response generated."
        extras = []
        if response.triage_result:
            extras.append(f"\n**Triage**: {response.triage_result.get('risk_level', 'N/A')}")
        if response.scheme_results:
            eligible = [r["scheme_id"] for r in response.scheme_results if r.get("eligible")]
            if eligible:
                extras.append(f"\n**Eligible Schemes**: {', '.join(eligible)}")
        if response.db_changes and response.db_changes.get("success"):
            extras.append(f"\n**DB Update**: {response.db_changes.get('db_action', '')}")
        history.append([text, reply + "".join(extras)])
        tts_audio = _wav_bytes_to_numpy(response.audio_bytes)
        return history, tts_audio

    def text_turn(message, language, history):
        """Handle typed text input."""
        history = [list(pair) for pair in history] if history else []
        if not message or not message.strip():
            return history, "", None
        try:
            lang_code = "hi" if language == "Hindi" else "en"
            history, tts_audio = _process_text(message.strip(), lang_code, history)
            return history, "", tts_audio
        except Exception as e:
            logger.exception("text_turn")
            history.append([message, f"**Error:** {e}"])
            return history, "", None

    def voice_turn_b64(audio_b64_str, language, history):
        """Handle voice input from the JS recorder.

        The browser records audio via Web Audio API, encodes it as a
        16-bit mono WAV, base64-encodes it, and stores it in a JS global.
        The button click's js= parameter injects it here.  Gradio never
        touches the audio — no gr.Audio component involved.
        """
        history = [list(pair) for pair in history] if history else []
        if not audio_b64_str:
            history.append(["🎤", "No audio recorded. Click Start Recording first."])
            return history, None
        try:
            audio_bytes = base64.b64decode(audio_b64_str)
            logger.info("Voice: decoded %d bytes from browser recording", len(audio_bytes))
            if len(audio_bytes) < 1000:
                history.append(["🎤 (audio)", "Recording too short. Please try again."])
                return history, None
            lang_code = "hi" if language == "Hindi" else "en"
            text = asr.transcribe(audio_bytes, lang_code)
            if not text or not text.strip():
                history.append(["🎤 (audio)", "Could not transcribe. Please speak clearly and try again."])
                return history, None
            history, tts_audio = _process_text(text.strip(), lang_code, history)
            if history:
                history[-1][0] = f"🎤 {text.strip()}"
            return history, tts_audio
        except Exception as e:
            logger.exception("voice_turn_b64")
            history.append(["🎤 (audio)", f"**Error:** {e}"])
            return history, None

    # ─────────────────────────────────────────────────────────────────────────

    def add_patient_voice(command, language):
        lang_code = "hi" if language == "Hindi" else "en"
        result = voice_db.process_voice_command(command, language=lang_code)
        output = f"**Intent**: {result.get('intent', 'unknown')}\n"
        output += f"**Success**: {result.get('success', False)}\n"
        output += f"**Action**: {result.get('db_action', 'None')}\n\n"
        if result.get("success"):
            output += f"**Response**: {result.get('message_hi', result.get('message_en', ''))}\n\n"
            output += f"**Entities**:\n```json\n{json.dumps(result.get('entities', {}), indent=2, default=str)}\n```"
        else:
            output += f"**Error**: {result.get('message_en', 'Unknown error')}"
        return output

    def get_recent_patients():
        if not spark:
            return []
        return spark.sql("""
            SELECT name, age, village, bpl_status, registered_date
            FROM workspace.asha_copilot.patients
            ORDER BY registered_date DESC LIMIT 10
        """).toPandas()

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
        out = f"## Risk Level: <span style='color:{color}'>{result.risk_level}</span>\n\n"
        out += f"**Method**: {result.method}\n\n### Alerts:\n"
        for a in result.alerts:
            out += f"- **{a.sign_name}** ({a.severity}): {a.action} [Urgency: {a.urgency_hours}h]\n"
        if result.assignment and result.assignment.phc_name:
            a = result.assignment
            out += f"\n### Referral:\n- **Facility**: {a.phc_name} ({a.phc_type})\n"
            out += f"- **Doctor**: {a.doctor_name} — {a.doctor_phone}\n"
        return out

    def check_schemes(name):
        if not name:
            return "Please enter a patient name."
        if not spark:
            return "Spark not available."
        patients = spark.sql(f"""
            SELECT * FROM workspace.asha_copilot.patients
            WHERE lower(name) LIKE '%{name.lower().split()[0]}%' LIMIT 1
        """).collect()
        if not patients:
            return f"Patient '{name}' not found."
        p = patients[0].asDict()
        results = scheme_engine.evaluate(p)
        out = f"## Scheme Eligibility — {p['name']}\n"
        for r in results:
            icon = "[Y]" if r.eligible else "[N]"
            out += f"### {icon} {r.scheme_id} — {r.scheme_name}\n"
            if r.amount:
                out += f"- **Amount**: Rs {r.amount:,}\n"
            if r.missing_requirements:
                out += f"- **Missing**: {', '.join(r.missing_requirements)}\n"
            out += f"- {r.notes}\n\n"
        return out

    def ask_protocol(question):
        if not question:
            return "Please enter a question."
        result = rag.answer(question)
        out = f"## Answer\n\n{result['answer']}\n\n"
        if result["sources"]:
            out += f"**Sources**: {', '.join(result['sources'])}"
        return out

    # ── UI ───────────────────────────────────────────────────────────────────

    with gr.Blocks(theme=gr.themes.Soft(), title="ASHA Copilot") as demo:
        gr.Markdown("# ASHA Copilot — AI-Powered Maternal & Child Healthcare Assistant")
        gr.Markdown("Voice-first, multilingual field assistant for India's community health workers")

        with gr.Tab("Voice Copilot"):
            gr.Markdown("### Chat with ASHA Copilot in Hindi or English")
            lang_select = gr.Radio(["Hindi", "English"], value="Hindi", label="Language")
            chatbot = gr.Chatbot(height=400)

            # ── Voice input (pure JS — bypasses gr.Audio entirely) ────────
            # The browser's Web Audio API captures 16 kHz mono PCM,
            # encodes it as a WAV, base64-encodes it, and stores it in
            # window.__asha_wav_b64.  The Send Voice button's js=
            # parameter injects that value into voice_turn_b64().
            gr.HTML(_RECORDER_HTML)
            audio_b64 = gr.Textbox(value="", visible=False)
            voice_btn = gr.Button("Send Voice", variant="primary")

            # ── Text input ────────────────────────────────────────────────
            gr.Markdown("#### Or type below")
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type ASHA's input here...", label="Text Input", scale=8
                )
                send_btn = gr.Button("Send", variant="secondary", scale=1)

            # ── TTS audio output ──────────────────────────────────────────
            tts_output = gr.Audio(
                label="Response Audio", autoplay=False,
                interactive=False,
            )

            # Voice: JS recorder → base64 injected via js= → Python handler
            voice_btn.click(
                voice_turn_b64,
                inputs=[audio_b64, lang_select, chatbot],
                outputs=[chatbot, tts_output],
                js="(b64, lang, hist) => [window.__asha_wav_b64 || '', lang, hist]",
            )
            # Text: typed input
            msg_input.submit(
                text_turn, [msg_input, lang_select, chatbot],
                [chatbot, msg_input, tts_output],
            )
            send_btn.click(
                text_turn, [msg_input, lang_select, chatbot],
                [chatbot, msg_input, tts_output],
            )

            gr.Examples(
                examples=[
                    ["Mujhe sab patients dikhao"],
                    ["Meena Kumari ka gaanv konsa hai?"],
                    ["Patient ko kal raat se severe headache aur chakkar aa raha hai. BP 160/100 tha."],
                    ["Naya patient register karo — Geeta Devi, umra 26, gaon Sarnath, BPL hai"],
                    ["Sunita Devi PMMVY scheme ke liye eligible hai kya?"],
                    ["BCG vaccine kab dena chahiye?"],
                ],
                inputs=msg_input,
            )

        with gr.Tab("Patient CRUD"):
            gr.Markdown("### Add/Update Patients via Natural Language")
            crud_lang = gr.Radio(["Hindi", "English"], value="Hindi", label="Language")
            crud_input = gr.Textbox(label="Voice Command", lines=2,
                placeholder="e.g., 'Naya patient register karo — Meena Kumari, umra 22, gaon Sultanpur'")
            crud_btn = gr.Button("Process Command", variant="primary")
            crud_output = gr.Markdown()
            crud_btn.click(add_patient_voice, [crud_input, crud_lang], crud_output)
            gr.Markdown("### Recent Patients")
            patients_table = gr.Dataframe(value=get_recent_patients, every=30)

        with gr.Tab("Triage"):
            gr.Markdown("### Clinical Triage — Enter Vitals")
            with gr.Row():
                bp_s = gr.Number(label="BP Systolic", value=120)
                bp_d = gr.Number(label="BP Diastolic", value=80)
                hb  = gr.Number(label="Hemoglobin (g/dL)", value=11.0)
                temp = gr.Number(label="Temperature (°C)", value=37.0)
            notes = gr.Textbox(label="Clinical Notes / Symptoms", lines=2)
            triage_btn = gr.Button("Run Triage", variant="primary")
            triage_output = gr.Markdown()
            triage_btn.click(run_triage, [bp_s, bp_d, hb, temp, notes], triage_output)

        with gr.Tab("Scheme Checker"):
            gr.Markdown("### Check Welfare Scheme Eligibility")
            scheme_name = gr.Textbox(label="Patient Name")
            scheme_btn = gr.Button("Check Eligibility", variant="primary")
            scheme_output = gr.Markdown()
            scheme_btn.click(check_schemes, scheme_name, scheme_output)

        with gr.Tab("Protocol Q&A"):
            gr.Markdown("### Ask Clinical Protocol Questions (RAG-powered)")
            qa_input = gr.Textbox(label="Question",
                placeholder="What are the danger signs during pregnancy?")
            qa_btn = gr.Button("Ask", variant="primary")
            qa_output = gr.Markdown()
            qa_btn.click(ask_protocol, qa_input, qa_output)
            gr.Examples(
                examples=[
                    ["What are the danger signs during pregnancy?"],
                    ["When should a newborn be referred to a health facility?"],
                    ["What is the treatment for severe anemia in pregnancy?"],
                ],
                inputs=qa_input,
            )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    _load_secrets()
    spark = _get_spark()
    demo = build_app(spark)
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()
