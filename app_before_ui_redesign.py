
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
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --background: #0a101a;
        --surface: #111c2b;
        --surface-2: #172438;
        --border: #26374d;
        --text: #e8eff8;
        --muted: #91a2b8;
        --accent: #54a9ed;
        --accent-light: #8bc9ff;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
    }

    [data-testid="stSidebar"] {
        background: #0d1725;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] h2 {
        font-family: 'Space Grotesk', sans-serif;
    }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 27px;
        font-weight: 700;
        letter-spacing: -1px;
        color: #f1f6fc;
    }

    .brand span {
        color: var(--accent);
    }

    .eyebrow {
        color: var(--accent-light);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero {
        background: linear-gradient(120deg, #14263b, #101b2b);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 25px 28px;
        margin-bottom: 22px;
    }

    .hero h1 {
        margin: 0;
        font-size: 30px;
    }

    .hero p {
        color: var(--muted);
        margin-top: 9px;
        margin-bottom: 0;
        font-size: 14px;
    }

    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 8px;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 24px;
        font-weight: 700;
        color: var(--text);
    }

    .muted {
        color: var(--muted);
        font-size: 13px;
    }

    .status-dot {
        display: inline-block;
        height: 8px;
        width: 8px;
        border-radius: 50%;
        background: #38c98a;
        margin-right: 7px;
    }

    .stButton > button {
        background: #287fc1;
        color: white;
        border: 1px solid #398fd1;
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    .stButton > button:hover {
        background: #3696dc;
        border-color: #65b6f5;
        color: white;
    }

    .stTextArea textarea, .stTextInput input {
        background: #101b2b !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 9px !important;
    }

    [data-testid="stFileUploader"] {
        background: #101b2b;
        border: 1px dashed #3b526e;
        border-radius: 12px;
        padding: 10px;
    }

    [data-testid="stImage"] {
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] {
        border-color: var(--border);
        border-radius: 10px;
    }

    .section-label {
        color: var(--accent-light);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    hr {
        border-color: var(--border);
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


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
        raise FileNotFoundError(
            f"Prompt file not found: {PROMPT_PATH}"
        )

    prompt = PROMPT_PATH.read_text(encoding="utf-8-sig").strip()

    if not prompt:
        raise ValueError("Prompt C v3 is empty.")

    return prompt


def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail(MAX_IMAGE_SIZE)

    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=90,
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


def ask_model(api_key, model, image_data_url, question):
    client = Groq(api_key=api_key)

    system_prompt = load_system_prompt()

    response = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                ],
            },
        ],
    )

    return (
        response.choices[0].message.content
        or "The model returned an empty response."
    )


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
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


# ============================================================
# SIDEBAR
# ============================================================
# ============================================================
# SIDEBAR
# ============================================================

# Process navigation requests before creating the radio widget.
if st.session_state.pop("navigate_to_workspace", False):
    st.session_state["workspace_page"] = "Image Workspace"

if st.session_state.pop("navigate_to_history", False):
    st.session_state["workspace_page"] = "Analysis History"

with st.sidebar:

    st.markdown(
        '<div class="brand">SATQUERY<span> AI</span></div>',
        unsafe_allow_html=True,
    )

    st.caption("GEOSPATIAL INTELLIGENCE PLATFORM")

    st.markdown("---")

    page = st.radio(
        "WORKSPACE",
        [
            "Dashboard",
            "Image Workspace",
            "Analysis History",
            "Settings",
        ],
        key="workspace_page",
        label_visibility="visible",
    )

    st.markdown("---")

    api_key = get_api_key()

    if api_key:
        st.markdown(
            '<span class="status-dot"></span> API configured',
            unsafe_allow_html=True,
        )
    else:
        st.warning("API key not configured")

    st.markdown("---")

    st.caption("MODEL CONFIGURATION")

    model = st.text_input(
        "Vision model ID",
        value=st.session_state.get(
            "selected_model",
            MODEL_DEFAULT,
        ),
        key="model_input",
    )

    st.session_state["selected_model"] = model.strip()

    st.markdown("---")

    st.caption("SATQUERY AI · DEVELOPMENT BUILD")
    st.caption("Evidence-grounded visual analysis")


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">REMOTE SENSING INTELLIGENCE</div>
        <h1>Geospatial Intelligence Workspace</h1>
        <p>
            Explore satellite and aerial imagery through
            evidence-grounded vision-language analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    history = load_history()
    st.session_state["history"] = history

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="panel">
            <div class="metric-label">PLATFORM STATUS</div>
            <div class="metric-value">Operational</div>
            <div class="muted">Application interface available</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="panel">
            <div class="metric-label">SAVED ANALYSES</div>
            <div class="metric-value">{len(history)}</div>
            <div class="muted">Locally stored records</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="panel">
            <div class="metric-label">VISION MODEL</div>
            <div class="metric-value">Qwen 3.8</div>
            <div class="muted">Groq API integration</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Start analysing")

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="panel">
            <div class="section-label">01 · Image Workspace</div>
            <p class="muted">
                Upload satellite or aerial imagery and
                inspect visible land-cover patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "Open Image Workspace",
            use_container_width=True,
        ):
            st.session_state["navigate_to_workspace"] = True
            st.rerun()

    with right:
        st.markdown("""
        <div class="panel">
            <div class="section-label">02 · Analysis History</div>
            <p class="muted">
                Review previous model responses and
                download saved analysis records.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(
            "View Analysis History",
            use_container_width=True,
        ):
            st.session_state["navigate_to_history"] = True
            st.rerun()

    st.markdown("### Recent activity")

    if history:
        for item in history[:5]:
            with st.container(border=True):
                st.write(item.get("filename", "Unknown image"))
                st.caption(
                    f'{item.get("timestamp", "")} · '
                    f'{item.get("model", "")}'
                )
                st.write(item.get("question", ""))

    else:
        st.info("No analyses yet. Upload an image to get started.")


# ============================================================
# IMAGE WORKSPACE
# ============================================================

elif page == "Image Workspace":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">IMAGE ANALYSIS MODULE</div>
        <h1>Image Workspace</h1>
        <p>
            Upload remote-sensing imagery and ask questions
            about visible spatial features.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload satellite / aerial imagery",
        type=ALLOWED_TYPES,
        help=f"Supported formats: PNG, JPG, JPEG, WEBP. Maximum size: {MAX_IMAGE_MB} MB.",
    )

    if uploaded is not None:

        if uploaded.size > MAX_IMAGE_MB * 1024 * 1024:
            st.error(
                f"Image exceeds the {MAX_IMAGE_MB} MB limit."
            )
            st.stop()

        try:
            image = Image.open(uploaded)
            image.load()
            image = ImageOps.exif_transpose(image)

        except Exception:
            st.error("Unable to open this image.")
            st.stop()

        st.markdown("---")

        left, right = st.columns([1.15, 1])

        with left:

            st.markdown(
                '<div class="section-label">IMAGE PREVIEW</div>',
                unsafe_allow_html=True,
            )

            st.image(
                image,
                use_container_width=True,
            )

            st.caption(
                f"{uploaded.name} · "
                f"{image.width} × {image.height} pixels · "
                f"{image.mode}"
            )

        with right:

            st.markdown(
                '<div class="section-label">AI ANALYSIS TERMINAL</div>',
                unsafe_allow_html=True,
            )

            question = st.text_area(
                "Enter your question",
                value=(
                    "Describe the main visible land-cover "
                    "features and their spatial arrangement."
                ),
                height=145,
                placeholder="Ask about roads, vegetation, water bodies, spatial patterns...",
            )

            analyze = st.button(
                "Run Image Analysis",
                type="primary",
                use_container_width=True,
            )

            if analyze:

                if not question.strip():
                    st.error("Please enter a question.")

                elif not api_key:
                    st.error(
                        "Configure your Groq API key before analysis."
                    )

                elif not model:
                    st.error("Enter a valid model ID.")

                else:

                    try:
                        with st.spinner(
                            "Inspecting imagery and generating analysis..."
                        ):

                            image_url = prepare_image(image)

                            answer = ask_model(
                                api_key,
                                model,
                                image_url,
                                question.strip(),
                            )

                        timestamp = datetime.now().astimezone().isoformat(
                            timespec="seconds"
                        )

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

                        st.success("Analysis completed successfully.")

                    except FileNotFoundError as exc:
                        st.error(str(exc))

                    except Exception as exc:
                        st.error(
                            "The model request failed. Check your API "
                            "key, model ID, network connection, and "
                            "Groq account limits."
                        )
                        st.caption(
                            f"Error type: {type(exc).__name__}"
                        )

        if st.session_state.get("last_answer"):

            st.markdown("---")
            st.markdown("### Analysis Results")

            st.markdown(
                st.session_state["last_answer"]
            )

            st.caption(
                "AI-generated interpretation. Verify important "
                "findings independently."
            )

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "Download TXT",
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
                    "Download JSON",
                    data=json.dumps(
                        export,
                        indent=4,
                        ensure_ascii=False,
                    ),
                    file_name="satquery_analysis.json",
                    mime="application/json",
                    use_container_width=True,
                )

    else:
        st.info(
            "Upload an image to activate the analysis workspace."
        )

        st.markdown("### Example analysis questions")

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

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">RECORD MANAGEMENT</div>
        <h1>Analysis History</h1>
        <p>
            Review and export previous SatQuery AI analyses.
        </p>
    </div>
    """, unsafe_allow_html=True)

    history = load_history()
    st.session_state["history"] = history

    if history:

        st.write(f"**{len(history)} saved analysis records**")

        st.download_button(
            "Export complete history (JSON)",
            data=json.dumps(
                history,
                indent=4,
                ensure_ascii=False,
            ),
            file_name="satquery_analysis_history.json",
            mime="application/json",
        )

        for index, item in enumerate(history):

            title = (
                f"{item.get('filename', 'Unknown image')} "
                f"· {item.get('timestamp', '')}"
            )

            with st.expander(title):

                st.caption(
                    f"Model: {item.get('model', 'Unknown')}"
                )

                st.markdown("**Question**")
                st.write(item.get("question", ""))

                st.markdown("**Analysis**")
                st.markdown(item.get("answer", ""))

                st.download_button(
                    "Download this analysis",
                    data=json.dumps(
                        item,
                        indent=4,
                        ensure_ascii=False,
                    ),
                    file_name=f"analysis_{index + 1}.json",
                    mime="application/json",
                    key=f"history_download_{index}",
                )

    else:
        st.info("No saved analyses are available yet.")


# ============================================================
# SETTINGS
# ============================================================

elif page == "Settings":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">SYSTEM CONFIGURATION</div>
        <h1>Settings</h1>
        <p>
            Application configuration and model information.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Model configuration")

    st.write(f"**Current model:** `{model}`")
    st.write(f"**Temperature:** `{TEMPERATURE}`")
    st.write(f"**Maximum output tokens:** `{MAX_TOKENS}`")

    st.markdown("---")

    st.markdown("### API configuration")

    if api_key:
        st.success("Groq API key detected.")
    else:
        st.error("Groq API key not detected.")

    st.caption(
        "API credentials are read from Streamlit secrets "
        "or the GROQ_API_KEY environment variable."
    )

    st.markdown("---")

    st.markdown("### Prompt configuration")

    if PROMPT_PATH.exists():

        prompt = load_system_prompt()

        st.success("Prompt C v3 loaded successfully.")

        st.write(f"**Prompt file:** `{PROMPT_PATH.name}`")
        st.write(f"**Prompt characters:** {len(prompt):,}")

        with st.expander("Preview active system prompt"):
            st.text(prompt)

    else:
        st.error("Prompt C v3 could not be found.")

    st.markdown("---")

    st.markdown("### Application limitations")

    st.info(
        "The current application performs image-based visual "
        "question answering. It does not yet perform pixel-level "
        "segmentation, georeferencing, validated object detection, "
        "or scientifically verified change detection."
    )

    st.markdown("---")

    st.markdown("### Local data storage")

    st.write(f"**History file:** `{HISTORY_PATH}`")

    st.caption(
        "History stores questions and model responses. "
        "Uploaded image files are not saved to the history database."
    )
