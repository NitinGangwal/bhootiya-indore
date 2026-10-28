"""
Bhootiya Indore: Dadi-Nani ki Kahaniyan
Offline oral-history tool: audio -> Whisper (local) -> Gemma via Ollama (local) -> spooky ofrenda cards.
Sab kuch apni machine par chalta hai, koi cloud API nahi.

Run = cd D:\Downloads; python -m streamlit run app.py
"""

import html
import json
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import requests
import streamlit as st

DATA_DIR = Path("stories")
DATA_DIR.mkdir(exist_ok=True)
OLLAMA_URL = "http://localhost:11434/api/generate"

st.set_page_config(page_title="Bhootiya Indore", page_icon="🕯️", layout="wide")

# ---------- Theme: raat + diye + genda phool ----------
st.markdown(
    """
<style>
.stApp { background: radial-gradient(circle at top, #1b1030 0%, #0b0613 70%); color: #f3e9d2; }
h1, h2, h3 { color: #ffb627 !important; font-family: Georgia, serif; }
.card { background: #150d22; border: 1px solid #ff8c1a55; border-radius: 14px;
        padding: 18px 20px; margin-bottom: 18px; box-shadow: 0 0 18px #ff8c1a22; }
.card h3 { margin: 0 0 4px 0; }
.meta { color: #c9b38a; font-size: 0.85rem; margin-bottom: 8px; }
.open { font-style: italic; color: #ffd27a; margin: 8px 0; }
.tag { display: inline-block; background: #ff8c1a22; color: #ffb627; border-radius: 10px;
       padding: 2px 10px; font-size: 0.8rem; margin-right: 6px; }
.fear { letter-spacing: 3px; }
.ofrenda { color: #ffb627; font-size: 0.9rem; margin-top: 10px; }

/* ---- Contrast fix: sab text saaf aur chamakdaar ---- */
html, body, .stApp { font-size: 17px; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp span,
.stMarkdown, [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p {
    color: #fff3da !important;
}
h1 { font-size: 2.6rem !important; text-shadow: 0 0 18px #ff8c1a88; }
h2, h3 { color: #ffb627 !important; }
[data-testid="stCaptionContainer"], .stApp small { color: #e6d2a8 !important; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #120a1f; border-right: 1px solid #ff8c1a44; }

/* tabs */
button[data-baseweb="tab"] p { color: #ffd27a !important; font-size: 1.1rem; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] p { color: #ffb627 !important; }
div[data-baseweb="tab-highlight"] { background-color: #ff8c1a !important; }

/* input boxes */
input, textarea, div[data-baseweb="select"] > div {
    background: #1f1433 !important; color: #fff3da !important;
    border: 1px solid #ff8c1a88 !important; border-radius: 8px !important;
}
textarea::placeholder, input::placeholder { color: #bba983 !important; }

/* upload box */
[data-testid="stFileUploaderDropzone"] { background: #1f1433 !important; border: 1px dashed #ff8c1a !important; }
[data-testid="stFileUploaderDropzone"] * { color: #fff3da !important; }

/* buttons */
.stButton > button, [data-testid="stFileUploaderDropzone"] button {
    background: #ff8c1a !important; color: #1a0d00 !important; font-weight: 700 !important;
    border: 0 !important; border-radius: 10px !important; padding: 0.5rem 1.2rem !important;
}
.stButton > button:hover { background: #ffb627 !important; box-shadow: 0 0 14px #ffb627aa; }
.stButton > button p { color: #1a0d00 !important; }
.stButton > button:disabled { background: #4a3a2a !important; }
.stButton > button:disabled p { color: #9c8a6c !important; }

/* cards */
.card { color: #fff3da; font-size: 1.02rem; line-height: 1.6; }
.card div { color: #fff3da; }
.meta, .meta span { color: #e6d2a8 !important; }
.open { color: #ffd27a !important; font-size: 1.1rem; }
.tag { color: #ffb627 !important; background: #ff8c1a33; }
.ofrenda { color: #ffb627 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Helpers ----------
@st.cache_resource(show_spinner="Whisper model load ho raha hai (pehli baar thoda time lagega)...")
def load_whisper(size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


def load_audio_16k(path: str):
    """Audio ko ffmpeg se 16kHz mono wav banakar numpy array deta hai.
    Isse faster-whisper ke `av` wale version-bug se bach jaate hain."""
    import subprocess
    import wave

    import imageio_ffmpeg
    import numpy as np

    wav_path = path + ".16k.wav"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", path, "-ac", "1", "-ar", "16000",
         "-acodec", "pcm_s16le", wav_path],
        check=True, capture_output=True,
    )
    with wave.open(wav_path, "rb") as w:
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(audio_path: str, size: str, progress=None) -> str:
    model = load_whisper(size)
    audio = load_audio_16k(audio_path)
    total = max(len(audio) / 16000, 1)
    segments, _info = model.transcribe(
        audio,
        language="hi",
        beam_size=1,                       # tez
        vad_filter=True,
        condition_on_previous_text=False,  # lambi audio mein repeat-loop se bachata hai
    )
    parts = []
    for seg in segments:
        parts.append(seg.text.strip())
        if progress is not None:
            progress.progress(min(seg.end / total, 1.0), text=f"{int(seg.end)} / {int(total)} second ho gaye")
    return " ".join(parts).strip()


PROMPT = """Tum ek lok-kathaon ke sangrahak ho. Neeche ek bujurg ne Indore ki ek bhootiya kahani sunayi hai.
Is transcript se (jo thoda galat ho sakta hai) ye JSON banao. Sirf JSON do, aur kuch nahi.

Keys:
- "title": kahani ka chhota darawna title (Hinglish, Roman script)
- "place": kahani kis jagah ki hai (agar transcript mein nahi hai to "Indore")
- "summary": 3-4 line ka summary, Hinglish (Roman script), spooky tone mein, sirf wahi likho jo transcript mein hai
- "fear_level": 1 se 5 tak ek number
- "opening_line": ek eerie, kavita jaisi opening line, Hinglish
- "remember": ek line jo us bujurg ki yaad mein ho, jaise "Dadi hamesha kehti thi ..." (sirf tab jab transcript mein ho, warna "")

Narrator: {narrator} ({relation})
Transcript:
\"\"\"{transcript}\"\"\"
"""


def ask_gemma(transcript: str, narrator: str, relation: str, model: str) -> dict:
    payload = {
        "model": model,
        "prompt": PROMPT.format(transcript=transcript, narrator=narrator, relation=relation),
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.7},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    raw = r.json()["response"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0)) if m else {}
    fear = data.get("fear_level", 3)
    try:
        fear = max(1, min(5, int(fear)))
    except (TypeError, ValueError):
        fear = 3
    return {
        "title": str(data.get("title", "Ek anjaani kahani")),
        "place": str(data.get("place", "Indore")),
        "summary": str(data.get("summary", "")),
        "fear_level": fear,
        "opening_line": str(data.get("opening_line", "")),
        "remember": str(data.get("remember", "")),
    }


def save_story(record: dict, audio_bytes: bytes | None, audio_ext: str):
    sid = record["id"]
    if audio_bytes:
        audio_name = f"{sid}{audio_ext}"
        (DATA_DIR / audio_name).write_bytes(audio_bytes)
        record["audio"] = audio_name
    (DATA_DIR / f"{sid}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def load_stories() -> list[dict]:
    items = []
    for p in DATA_DIR.glob("*.json"):
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return sorted(items, key=lambda x: x.get("created", 0), reverse=True)


def esc(x) -> str:
    return html.escape(str(x))


# ---------- Sidebar ----------
with st.sidebar:
    st.title("🕯️ Settings")
    whisper_size = st.selectbox("Whisper size", ["small", "base", "medium"], index=0,
                                help="small = tez, medium = behtar Hindi par dheema")
    gemma_model = st.text_input("Ollama model", value="gemma3:4b")
    st.caption("Sab kuch local chalta hai. Internet band karke bhi try karo.")

st.title("🕯️ Bhootiya Indore")
st.caption("Dadi-Nani ki kahaniyan, jo kisi server par nahi, sirf hamari yaadon mein rehni chahiye.")

tab_add, tab_gallery = st.tabs(["➕ Nayi kahani jodo", "👻 Haunted Indore ki kahaniyan"])

# ---------- Tab 1: add ----------
with tab_add:
    c1, c2 = st.columns(2)
    narrator = c1.text_input("Kahani kisne sunayi? (naam)", value="Dadi")
    relation = c2.text_input("Rishta", value="Dadi ji")
    source = st.radio("Kahani ka source", ["Asli (ghar ke bade ki awaaz)", "Retold (maine apne shabdon mein sunayi)"],
                      horizontal=True)

    audio = st.file_uploader("Audio upload karo (mp3, wav, m4a, ogg, opus)",
                             type=["mp3", "wav", "m4a", "ogg", "opus", "aac"])
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""

    if audio and st.button("🎙️ Transcribe karo (offline)"):
        suffix = Path(audio.name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio.getvalue())
            tmp_path = tmp.name
        t0 = time.time()
        bar = st.progress(0.0, text="Whisper sun raha hai...")
        st.session_state.transcript = transcribe(tmp_path, whisper_size, bar)
        bar.empty()
        st.success(f"Ho gaya ({time.time() - t0:.0f} sec). Neeche transcript theek kar lo.")

    transcript = st.text_area(
        "Transcript (Whisper se galtiyan ho sakti hain, yahan edit kar lo. Bina audio ke bhi yahan seedha text likh sakte ho)",
        key="transcript", height=220)

    if st.button("🕯️ Kahani ko bhootiya card banao", type="primary", disabled=not transcript.strip()):
        try:
            with st.spinner("Gemma kahani ko aakaar de raha hai..."):
                info = ask_gemma(transcript, narrator, relation, gemma_model)
            record = {
                "id": uuid.uuid4().hex[:8],
                "created": time.time(),
                "narrator": narrator,
                "relation": relation,
                "source": "asli" if source.startswith("Asli") else "retold",
                "transcript": transcript,
                **info,
            }
            ext = Path(audio.name).suffix if audio else ""
            save_story(record, audio.getvalue() if audio else None, ext)
            st.success("Kahani ofrenda mein jud gayi! 'Haunted Indore' tab dekho.")
        except requests.exceptions.ConnectionError:
            st.error("Ollama chal nahi raha. Terminal mein `ollama serve` chalao aur model pull karo.")
        except Exception as e:
            st.error(f"Kuch gadbad hui: {e}")

# ---------- Tab 2: gallery ----------
with tab_gallery:
    stories = load_stories()
    if not stories:
        st.info("Abhi koi kahani nahi hai. Pehle ek kahani jodo 🕯️")
    n_asli = sum(1 for s in stories if s.get("source") == "asli")
    if stories:
        st.markdown(f"**{len(stories)} kahaniyan** · {n_asli} asli awaaz mein · baaki retold")
    for s in stories:
        fear = "💀" * s["fear_level"] + "🕯️" * (5 - s["fear_level"])
        label = "Asli awaaz" if s.get("source") == "asli" else "Retold"
        st.markdown(
            f"""
<div class="card">
  <h3>{esc(s['title'])}</h3>
  <div class="meta">📍 {esc(s['place'])} &nbsp;·&nbsp; <span class="tag">{label}</span>
  <span class="fear">{fear}</span></div>
  <div class="open">“{esc(s['opening_line'])}”</div>
  <div>{esc(s['summary'])}</div>
  <div class="ofrenda">🌼 Suna: {esc(s['narrator'])} ({esc(s['relation'])}) se
  {(' · ' + esc(s['remember'])) if s.get('remember') else ''}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        if s.get("audio") and (DATA_DIR / s["audio"]).exists():
            st.audio(str(DATA_DIR / s["audio"]))
        with st.expander("Poora transcript"):
            st.write(s["transcript"])
            
            
            
            
            
            
            
            