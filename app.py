import base64
import io
import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================
ROOT = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "prompts" / "prompt_C_v3_production.txt"
DATA_DIR = ROOT / "data"
HISTORY_PATH = DATA_DIR / "analysis_history.json"

MODEL_DEFAULT = "qwen/qwen3.8-27b"
MAX_IMAGE_MB = 10
MAX_IMAGE_SIZE = (1600, 1600)
MAX_TOKENS = 900
TEMPERATURE = 0.2
ALLOWED_TYPES = ["png", "jpg", "jpeg", "webp"]

st.set_page_config(
    page_title="SatQuery AI | Earth Observation",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {
  --bg: #07111d;
  --bg2: #0a1725;
  --panel: rgba(13, 29, 46, .88);
  --panel2: #10243a;
  --line: rgba(112, 164, 197, .20);
  --text: #e8f3fb;
  --muted: #91a9bd;
  --cyan: #58dcf2;
  --blue: #56a8ff;
  --teal: #4de0bd;
  --amber: #f2c879;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
  color: var(--text);
  background:
    radial-gradient(ellipse at 78% 4%, rgba(29, 106, 145, .20), transparent 31%),
    radial-gradient(ellipse at 6% 28%, rgba(21, 75, 111, .15), transparent 34%),
    linear-gradient(145deg, #060e18 0%, #091827 52%, #07111d 100%);
}
.stApp:before {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  opacity: .12;
  background-image: linear-gradient(rgba(91, 169, 204, .13) 1px, transparent 1px), linear-gradient(90deg, rgba(91, 169, 204, .13) 1px, transparent 1px);
  background-size: 56px 56px;
  mask-image: linear-gradient(to bottom, black, transparent 82%);
}
[data-testid="stAppViewContainer"] > .main { position: relative; z-index: 1; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important; letter-spacing: -.035em; }
p, label, .stMarkdown { color: var(--text); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #091827 0%, #0a1b2b 55%, #081522 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
.brand-lockup { padding: 12px 4px 18px; }
.brand-mark { display:flex; align-items:center; gap:10px; }
.brand-icon {
  width:42px;height:42px;border-radius:13px;display:grid;place-items:center;
  background:linear-gradient(145deg,rgba(88,220,242,.20),rgba(86,168,255,.08));
  border:1px solid rgba(88,220,242,.35);font-size:22px;box-shadow:0 0 26px rgba(88,220,242,.10);
}
.brand-name {font-family:'Space Grotesk';font-size:25px;font-weight:700;letter-spacing:-1.2px;color:#f1f8ff;line-height:1;}
.brand-name span {color:var(--cyan)}
.brand-sub {font:10px 'DM Mono',monospace;letter-spacing:1.7px;color:#7797ad;margin:10px 0 0 53px;}
.side-label,.eyebrow {font:500 10px 'DM Mono',monospace;letter-spacing:2px;text-transform:uppercase;color:var(--cyan);}
.side-status {display:flex;align-items:center;gap:9px;padding:11px 12px;border:1px solid var(--line);background:rgba(10,29,44,.75);border-radius:10px;font-size:12px;color:#c5d8e7;}
.pulse {width:7px;height:7px;border-radius:50%;background:var(--teal);box-shadow:0 0 12px var(--teal);}
.hero {
  position:relative;overflow:hidden;padding:30px 32px;border-radius:20px;margin-bottom:22px;
  border:1px solid rgba(103,174,207,.28);
  background:linear-gradient(112deg,rgba(16,43,65,.97),rgba(10,28,45,.92) 58%,rgba(13,47,63,.83));
  box-shadow:0 18px 55px rgba(0,0,0,.18);
}
.hero:after {content:"";position:absolute;width:330px;height:330px;right:5%;top:-180px;border-radius:50%;border:1px solid rgba(88,220,242,.22);box-shadow:0 0 0 28px rgba(88,220,242,.025),0 0 0 58px rgba(88,220,242,.02),0 0 80px rgba(88,220,242,.10);}
.hero-content {position:relative;z-index:1;max-width:760px;}
.hero h1 {font-size:clamp(30px,4vw,46px);line-height:1.08;margin:10px 0 12px;}
.hero p {color:#a9c1d2;font-size:14px;max-width:680px;margin:0;line-height:1.7;}
.hero-meta {display:flex;gap:9px;flex-wrap:wrap;margin-top:22px;}
.meta-chip {font:10px 'DM Mono',monospace;letter-spacing:.8px;color:#b8d9e8;border:1px solid rgba(113,185,213,.23);background:rgba(4,18,30,.38);padding:7px 10px;border-radius:6px;}
.panel {background:linear-gradient(155deg,rgba(16,37,57,.94),rgba(10,25,40,.94));border:1px solid var(--line);border-radius:15px;padding:19px 20px;box-shadow:0 12px 32px rgba(0,0,0,.12);}
.panel-kicker {font:10px 'DM Mono',monospace;letter-spacing:1.6px;color:#78a7c1;text-transform:uppercase;margin-bottom:10px;}
.metric-value {font:600 27px 'Space Grotesk',sans-serif;color:#effaff;letter-spacing:-1px;}
.metric-caption {font-size:11px;color:#87a5ba;margin-top:5px;}
.metric-accent {color:var(--cyan);}
.section-heading {display:flex;align-items:center;gap:10px;margin:26px 0 13px;font:600 17px 'Space Grotesk',sans-serif;color:#eaf6ff;}
.section-index {font:11px 'DM Mono',monospace;color:var(--cyan);border:1px solid rgba(88,220,242,.3);padding:4px 7px;border-radius:5px;}
.feature-card {height:100%;background:linear-gradient(145deg,rgba(16,39,59,.92),rgba(10,25,41,.92));border:1px solid var(--line);border-radius:14px;padding:19px;}
.feature-icon {font-size:21px;margin-bottom:12px;}
.feature-title {font:600 15px 'Space Grotesk',sans-serif;color:#eaf6ff;margin-bottom:7px;}
.feature-copy {font-size:12px;line-height:1.6;color:#8faabe;}
.terminal-head {display:flex;justify-content:space-between;align-items:center;padding:11px 14px;background:#0a1b2b;border:1px solid var(--line);border-radius:10px 10px 0 0;font:10px 'DM Mono',monospace;color:#8eb5cb;letter-spacing:1px;}
.terminal-body {padding:15px;background:rgba(5,17,28,.72);border:1px solid var(--line);border-top:0;border-radius:0 0 10px 10px;}
.status-pill {display:inline-flex;align-items:center;gap:7px;border:1px solid rgba(77,224,189,.25);background:rgba(77,224,189,.07);padding:6px 9px;border-radius:30px;color:#8be8d1;font:10px 'DM Mono',monospace;letter-spacing:.8px;}
.small-mono {font:10px 'DM Mono',monospace;color:#7696ac;letter-spacing:1px;}
div[data-testid="stFileUploader"] {background:rgba(8,25,40,.7);border:1px dashed rgba(88,220,242,.38);border-radius:13px;padding:12px;}
div[data-testid="stFileUploader"] section {background:transparent;}
.stTextArea textarea,.stTextInput input {background:#091a29!important;color:#e9f6ff!important;border:1px solid rgba(106,163,192,.28)!important;border-radius:9px!important;}
.stTextArea textarea:focus,.stTextInput input:focus {border-color:var(--cyan)!important;box-shadow:0 0 0 1px rgba(88,220,242,.15)!important;}
.stButton > button,.stDownloadButton > button {border-radius:9px;border:1px solid rgba(88,220,242,.42);background:linear-gradient(110deg,#12627b,#167da0);color:#f2fcff;font-weight:600;min-height:42px;transition:all .18s ease;}
.stButton > button:hover,.stDownloadButton > button:hover {border-color:var(--cyan);background:linear-gradient(110deg,#167a94,#1599b8);color:white;box-shadow:0 0 20px rgba(88,220,242,.13);}
.stButton > button[kind="secondary"] {background:rgba(15,36,54,.85);border:1px solid var(--line);}
[data-testid="stImage"] img {border:1px solid var(--line);border-radius:12px;}
div[data-testid="stExpander"] {background:rgba(12,29,45,.72);border:1px solid var(--line);border-radius:11px;}
div[data-testid="stExpander"] summary {color:#d9edf8;}
.stTabs [data-baseweb="tab-list"] {gap:8px;background:transparent;}
.stTabs [data-baseweb="tab"] {background:rgba(12,29,45,.7);border:1px solid var(--line);border-radius:8px 8px 0 0;padding:9px 15px;}
.stTabs [aria-selected="true"] {border-color:rgba(88,220,242,.5)!important;color:var(--cyan)!important;}
hr {border-color:var(--line);}
[data-testid="stAlert"] {border-radius:10px;}
footer {visibility:hidden;}
#MainMenu {visibility:hidden;}
@media(max-width:800px){.hero{padding:23px 20px}.hero:after{right:-190px}.block-container{padding-left:1rem;padding-right:1rem}}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA AND API HELPERS
# ============================================================
def get_api_key():
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GROQ_API_KEY", "")


def load_system_prompt():
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    prompt = PROMPT_PATH.read_text(encoding="utf-8-sig").strip()
    if not prompt:
        raise ValueError("Prompt C v3 is empty.")
    return prompt


def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail(MAX_IMAGE_SIZE)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def ask_model(api_key, model, image_data_url, question):
    client = Groq(api_key=api_key)
    system_prompt = load_system_prompt()
    response = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    )
    return response.choices[0].message.content or "The model returned an empty response."


def load_history():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        return []
    try:
        with HISTORY_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_history(record):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()
    history.insert(0, record)
    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, ensure_ascii=False)


def initialize_state():
    defaults = {
        "history": load_history(),
        "last_answer": None,
        "last_question": None,
        "last_filename": None,
        "last_timestamp": None,
        "last_model": None,
        "selected_model": MODEL_DEFAULT,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def hero(eyebrow, title, description, chips=None):
    chip_html = "".join(f'<span class="meta-chip">{chip}</span>' for chip in (chips or []))
    st.markdown(
        f"""
        <div class="hero"><div class="hero-content">
          <div class="eyebrow">{eyebrow}</div>
          <h1>{title}</h1><p>{description}</p>
          <div class="hero-meta">{chip_html}</div>
        </div></div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(index, title):
    st.markdown(
        f'<div class="section-heading"><span class="section-index">{index}</span>{title}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, caption, accent=False):
    value_class = "metric-value metric-accent" if accent else "metric-value"
    st.markdown(
        f"""<div class="panel"><div class="panel-kicker">{label}</div>
        <div class="{value_class}">{value}</div><div class="metric-caption">{caption}</div></div>""",
        unsafe_allow_html=True,
    )


initialize_state()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
if st.session_state.pop("navigate_to_workspace", False):
    st.session_state["workspace_page"] = "Image Workspace"
if st.session_state.pop("navigate_to_history", False):
    st.session_state["workspace_page"] = "Analysis History"

api_key = get_api_key()
with st.sidebar:
    st.markdown(
        """<div class="brand-lockup"><div class="brand-mark"><div class="brand-icon">🛰️</div>
        <div class="brand-name">SATQUERY<span> AI</span></div></div>
        <div class="brand-sub">EARTH OBSERVATION SYSTEMS</div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="side-label">MISSION CONTROL</div>', unsafe_allow_html=True)
    page = st.radio(
        "Workspace navigation",
        ["Dashboard", "Image Workspace", "Analysis History", "Settings"],
        key="workspace_page",
        label_visibility="collapsed",
    )
    st.markdown("<br>", unsafe_allow_html=True)
    if api_key:
        st.markdown('<div class="side-status"><span class="pulse"></span> GROQ API CONNECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="side-status">⚠ API KEY NOT CONFIGURED</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="side-label">MODEL CONFIGURATION</div>', unsafe_allow_html=True)
    model = st.text_input(
        "Vision model ID",
        value=st.session_state.get("selected_model", MODEL_DEFAULT),
        key="model_input",
        help="Groq vision model identifier.",
    )
    model = model.strip()
    st.session_state["selected_model"] = model
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="small-mono">SATQUERY AI / BUILD 01.26</div>', unsafe_allow_html=True)
    st.caption("Evidence-grounded multimodal remote-sensing analysis")


# ============================================================
# DASHBOARD
# ============================================================
if page == "Dashboard":
    history = load_history()
    st.session_state["history"] = history
    hero(
        "SATELLITE VISION · GEOSPATIAL INTELLIGENCE",
        "See Earth from a different perspective.",
        "An interactive vision-language workspace for exploring satellite and aerial imagery, interpreting visible land-cover features, and asking evidence-grounded questions.",
        ["VISION-LANGUAGE ANALYSIS", "REMOTE SENSING", "EVIDENCE-GROUNDED"],
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("PLATFORM STATUS", "ONLINE", "Application interface operational", True)
    with c2:
        metric_card("SAVED ANALYSES", f"{len(history):02d}", "Locally stored observations")
    with c3:
        metric_card("VISION ENGINE", "QWEN 3.8", "Powered through Groq API")
    with c4:
        metric_card("ANALYSIS MODE", "VQA", "Visual question answering")

    section_heading("01", "Launch a mission")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(
            """<div class="feature-card"><div class="feature-icon">🛰️</div>
            <div class="feature-title">Satellite Image Workspace</div>
            <div class="feature-copy">Load a satellite or aerial image, inspect its visual characteristics, and submit natural-language questions to the vision model.</div></div>""",
            unsafe_allow_html=True,
        )
        if st.button("OPEN IMAGE WORKSPACE  ↗", use_container_width=True):
            st.session_state["navigate_to_workspace"] = True
            st.rerun()
    with b:
        st.markdown(
            """<div class="feature-card"><div class="feature-icon">📡</div>
            <div class="feature-title">Observation Archive</div>
            <div class="feature-copy">Revisit saved questions and model interpretations. Export individual observations or the complete analysis history.</div></div>""",
            unsafe_allow_html=True,
        )
        if st.button("VIEW ANALYSIS HISTORY  ↗", use_container_width=True):
            st.session_state["navigate_to_history"] = True
            st.rerun()

    section_heading("02", "Recent observations")
    if history:
        for item in history[:4]:
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"**{item.get('filename', 'Unknown image')}**")
                    st.caption(f"{item.get('timestamp', '')}  ·  {item.get('model', '')}")
                    st.write(item.get("question", ""))
                with right:
                    st.markdown('<div class="small-mono">ARCHIVED</div>', unsafe_allow_html=True)
    else:
        st.info("No observations yet. Open the Image Workspace to begin your first analysis.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("SATQUERY AI · AI-GENERATED INTERPRETATIONS SHOULD BE VERIFIED BEFORE OPERATIONAL USE.")


# ============================================================
# IMAGE WORKSPACE
# ============================================================
elif page == "Image Workspace":
    hero(
        "MISSION MODULE 01 · IMAGE INTERROGATION",
        "Satellite image workspace",
        "Upload remote-sensing imagery, inspect the scene, and query the vision model. Keep interpretations tied to what is actually visible in the image.",
        ["PNG / JPG / WEBP", f"MAX {MAX_IMAGE_MB} MB", "QWEN VISION"],
    )
    uploaded = st.file_uploader(
        "SELECT IMAGE SOURCE",
        type=ALLOWED_TYPES,
        help=f"Supported formats: PNG, JPG, JPEG, WEBP. Maximum size: {MAX_IMAGE_MB} MB.",
    )
    if uploaded is not None:
        if uploaded.size > MAX_IMAGE_MB * 1024 * 1024:
            st.error(f"Image exceeds the {MAX_IMAGE_MB} MB limit.")
            st.stop()
        try:
            image = Image.open(uploaded)
            image.load()
            image = ImageOps.exif_transpose(image)
        except Exception:
            st.error("Unable to open this image. Please try another supported image file.")
            st.stop()

        section_heading("01", "Scene inspection")
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.markdown('<div class="panel-kicker">SOURCE IMAGERY / PREVIEW</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="small-mono">WIDTH<br><b>{image.width}px</b></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="small-mono">HEIGHT<br><b>{image.height}px</b></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="small-mono">FORMAT<br><b>{image.format or uploaded.type or "IMAGE"}</b></div>', unsafe_allow_html=True)
            st.caption(f"Source file: {uploaded.name} · {uploaded.size / (1024 * 1024):.2f} MB")
        with right:
            st.markdown(
                '<div class="terminal-head"><span>⌁ ANALYSIS TERMINAL</span><span>INPUT READY</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="terminal-body">', unsafe_allow_html=True)
            st.markdown('<div class="small-mono">QUERY THE OBSERVED SCENE</div>', unsafe_allow_html=True)
            question = st.text_area(
                "Enter your question",
                value="Describe the main visible land-cover features and their spatial arrangement.",
                height=155,
                placeholder="Ask about roads, vegetation, water bodies, built-up areas, or spatial patterns...",
                label_visibility="collapsed",
            )
            st.markdown(
                '<div class="small-mono">TIP · ASK ABOUT VISIBLE FEATURES; AVOID ASSUMING SENSOR OR LOCATION DETAILS.</div>',
                unsafe_allow_html=True,
            )
            analyze = st.button("▶  RUN IMAGE ANALYSIS", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if analyze:
                if not question.strip():
                    st.error("Please enter a question.")
                elif not api_key:
                    st.error("Configure your Groq API key before analysis.")
                elif not model:
                    st.error("Enter a valid model ID.")
                else:
                    try:
                        with st.spinner("Interrogating image · generating evidence-grounded response..."):
                            image_url = prepare_image(image)
                            answer = ask_model(api_key, model, image_url, question.strip())
                        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
                        record = {
                            "timestamp": timestamp,
                            "filename": uploaded.name,
                            "question": question.strip(),
                            "answer": answer,
                            "model": model,
                        }
                        save_history(record)
                        st.session_state["history"] = load_history()
                        st.session_state["last_answer"] = answer
                        st.session_state["last_question"] = question.strip()
                        st.session_state["last_filename"] = uploaded.name
                        st.session_state["last_timestamp"] = timestamp
                        st.session_state["last_model"] = model
                        st.success("Analysis completed and observation archived.")
                    except FileNotFoundError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error("The model request failed. Check your API key, model ID, network connection, and Groq account limits.")
                        st.caption(f"Error type: {type(exc).__name__}")

        if st.session_state.get("last_answer"):
            section_heading("02", "Analysis output")
            st.markdown(
                '<div class="terminal-head"><span>⌁ MODEL RESPONSE</span><span class="status-pill"><span class="pulse"></span> RESPONSE RECEIVED</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="terminal-body">', unsafe_allow_html=True)
            st.markdown(st.session_state["last_answer"])
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption(
                f"Image: {st.session_state['last_filename']}  ·  Model: {st.session_state['last_model']}  ·  {st.session_state['last_timestamp']}"
            )
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "DOWNLOAD REPORT · TXT",
                    data=(
                        "SATQUERY AI ANALYSIS\n\n"
                        f"Image: {st.session_state['last_filename']}\n"
                        f"Question: {st.session_state['last_question']}\n\n"
                        f"{st.session_state['last_answer']}"
                    ),
                    file_name="satquery_analysis.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with col2:
                export = {
                    "image": st.session_state["last_filename"],
                    "question": st.session_state["last_question"],
                    "answer": st.session_state["last_answer"],
                    "model": st.session_state["last_model"],
                    "timestamp": st.session_state["last_timestamp"],
                }
                st.download_button(
                    "EXPORT STRUCTURED DATA · JSON",
                    data=json.dumps(export, indent=4, ensure_ascii=False),
                    file_name="satquery_analysis.json",
                    mime="application/json",
                    use_container_width=True,
                )
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        section_heading("01", "Ready for image input")
        st.markdown(
            """<div class="feature-card"><div class="feature-icon">🌐</div>
            <div class="feature-title">Awaiting satellite or aerial imagery</div>
            <div class="feature-copy">Select an image above to activate the scene preview and analysis terminal. The application accepts PNG, JPG, JPEG, and WEBP files up to 10 MB.</div></div>""",
            unsafe_allow_html=True,
        )
        section_heading("02", "Suggested observation queries")
        examples = [
            "What are the main visible land-cover features?",
            "Describe the spatial arrangement of visible objects.",
            "Where are roads, waterways, or linear structures visible?",
            "What can and cannot be reliably concluded from this image?",
        ]
        for example in examples:
            st.markdown(f"- {example}")


# ============================================================
# ANALYSIS HISTORY
# ============================================================
elif page == "Analysis History":
    hero(
        "MISSION MODULE 02 · OBSERVATION ARCHIVE",
        "Analysis history",
        "Review previous image interrogations, revisit model responses, and export your saved observation records.",
        ["LOCAL RECORDS", "JSON EXPORT", "SESSION ARCHIVE"],
    )
    history = load_history()
    st.session_state["history"] = history
    if history:
        top1, top2 = st.columns([3, 1])
        with top1:
            st.markdown(f"<div class='metric-value'>{len(history):02d} <span style='font-size:15px;color:#8daabd'>SAVED OBSERVATIONS</span></div>", unsafe_allow_html=True)
        with top2:
            st.download_button(
                "EXPORT FULL ARCHIVE ↓",
                data=json.dumps(history, indent=4, ensure_ascii=False),
                file_name="satquery_analysis_history.json",
                mime="application/json",
                use_container_width=True,
            )
        section_heading("01", "Archived observations")
        for index, item in enumerate(history):
            filename = item.get("filename", "Unknown image")
            timestamp = item.get("timestamp", "")
            with st.expander(f"{index + 1:02d}  ·  {filename}  ·  {timestamp}"):
                left, right = st.columns([3, 1])
                with left:
                    st.markdown('<div class="small-mono">OBSERVATION QUERY</div>', unsafe_allow_html=True)
                    st.write(item.get("question", ""))
                    st.markdown('<div class="small-mono">MODEL INTERPRETATION</div>', unsafe_allow_html=True)
                    st.markdown(item.get("answer", ""))
                with right:
                    st.markdown('<div class="small-mono">MODEL</div>', unsafe_allow_html=True)
                    st.write(item.get("model", "Unknown"))
                    st.download_button(
                        "DOWNLOAD RECORD",
                        data=json.dumps(item, indent=4, ensure_ascii=False),
                        file_name=f"satquery_observation_{index + 1}.json",
                        mime="application/json",
                        key=f"history_download_{index}",
                        use_container_width=True,
                    )
    else:
        st.info("No saved observations are available yet. Complete an image analysis to create your first record.")


# ============================================================
# SETTINGS
# ============================================================
elif page == "Settings":
    hero(
        "MISSION MODULE 03 · SYSTEM CONFIGURATION",
        "Platform settings",
        "Review the active vision model, API connection, production prompt, and current system capabilities.",
        ["MODEL CONTROL", "API STATUS", "PROMPT STATUS"],
    )
    section_heading("01", "Inference configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("ACTIVE VISION MODEL", model or "NOT SET", "Configured in the sidebar", True)
    with c2:
        metric_card("TEMPERATURE", str(TEMPERATURE), "Generation parameter")
    with c3:
        metric_card("MAX OUTPUT", f"{MAX_TOKENS:,} TOKENS", "Response limit")

    section_heading("02", "Service connectivity")
    if api_key:
        st.success("Groq API credentials detected. The application is configured to submit model requests.")
    else:
        st.error("Groq API key not detected. Add GROQ_API_KEY to Streamlit secrets or environment variables.")
    st.caption("Credentials are read from .streamlit/secrets.toml or the GROQ_API_KEY environment variable. Never commit your actual secrets file.")

    section_heading("03", "Production prompt")
    if PROMPT_PATH.exists():
        try:
            prompt = load_system_prompt()
            p1, p2 = st.columns(2)
            with p1:
                st.markdown('<div class="panel"><div class="panel-kicker">PROMPT STATUS</div><div class="metric-value metric-accent">LOADED</div><div class="metric-caption">Production prompt C v3 available</div></div>', unsafe_allow_html=True)
            with p2:
                st.markdown(f'<div class="panel"><div class="panel-kicker">PROMPT LENGTH</div><div class="metric-value">{len(prompt):,}</div><div class="metric-caption">Characters · UTF-8 text</div></div>', unsafe_allow_html=True)
            with st.expander("VIEW ACTIVE SYSTEM PROMPT"):
                st.text(prompt)
        except Exception as exc:
            st.error(f"Prompt could not be loaded: {exc}")
    else:
        st.error(f"Production prompt file not found: {PROMPT_PATH}")

    section_heading("04", "Current system scope")
    st.info(
        "SatQuery AI currently performs image-based visual question answering. It does not yet perform pixel-level segmentation, georeferencing, validated object detection, or scientifically verified change detection. Model interpretations should be independently verified before operational use."
    )
    st.caption(f"Local analysis history: {HISTORY_PATH} · Uploaded image files are not stored in the history database.")


# ============================================================
# FOOTER
# ============================================================
st.markdown("<br><hr>", unsafe_allow_html=True)
left, right = st.columns([3, 1])
with left:
    st.markdown('<div class="small-mono">SATQUERY AI · EARTH OBSERVATION INTELLIGENCE</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="small-mono" style="text-align:right">POWERED BY GROQ + QWEN VISION</div>', unsafe_allow_html=True)
