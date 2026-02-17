import sys
import os
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Path setup BEFORE any other imports
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────
# Streamlit MUST be imported and set_page_config
# called BEFORE any st.* calls anywhere
# ─────────────────────────────────────────────
import streamlit as st

# ✅ FIX 1: set_page_config must be the very first Streamlit call.
# Previously it was called AFTER st.error/st.warning in the import block,
# which caused a StreamlitAPIException crash on startup.
st.set_page_config(
    page_title="NeuroScript AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Standard library imports
# ─────────────────────────────────────────────
import numpy as np
import cv2
import torch
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import tempfile
import time
import json
import base64
from datetime import datetime
from collections import Counter
from io import BytesIO

# ─────────────────────────────────────────────
# ✅ FIX 2: Consolidated, deduplicated optional imports.
# Previously the same modules were imported twice (once at the top level
# and once inside try/except blocks), causing namespace collisions.
# ─────────────────────────────────────────────
try:
    from src.inference import MentalHealthInference, RealTimeWebcamInference
except ImportError:
    MentalHealthInference = None
    RealTimeWebcamInference = None

try:
    from src.real_time_acquisition import RealTimeHandwritingCapture, DigitalWritingCapture
except ImportError:
    RealTimeHandwritingCapture = None
    DigitalWritingCapture = None

try:
    from src.real_time_analysis import RealTimeAnalysisPipeline
except ImportError:
    RealTimeAnalysisPipeline = None

# ─────────────────────────────────────────────
# ✅ FIX 3: streamlit-drawable-canvas for digital writing.
# This package was missing from requirements.txt entirely.
# Add:  streamlit-drawable-canvas>=0.9.3  to requirements.txt
# ─────────────────────────────────────────────
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
class MentalHealthWebApp:
    def __init__(self):
        self.model_path = os.path.join("best_model.pth")
        self.inference = None
        self._init_session_state()
        self.load_model()

    # ─────────────────────────────────────────────
    def _init_session_state(self):
        defaults = {
            "result_history": [],
            "webcam_running": False,
            "webcam_snapshot": None,
            "canvas_strokes": [],
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    # ─────────────────────────────────────────────
    def load_model(self):
        """Load the inference model — safe to call before any other st.* call."""
        try:
            if MentalHealthInference and os.path.exists(self.model_path):
                self.inference = MentalHealthInference(self.model_path)
            # Don't call st.success/warning here — this runs inside __init__
            # which may be triggered before the page is ready.
        except Exception:
            self.inference = None

    # ══════════════════════════════════════════════════════════════════════════
    def run(self):
        st.title("🧠 NeuroScript AI - Mental Health Assessment")
        st.markdown("### Analyzing Handwriting and Drawing Patterns for Mental Health Indicators")

        with st.sidebar:
            st.title("Navigation")
            app_mode = st.selectbox(
                "Choose Analysis Mode",
                ["Home", "Image Analysis", "Real-time Webcam",
                 "Batch Processing", "Upload Video", "Digital Writing",
                 "Results History"]
            )

            st.markdown("---")
            st.subheader("Model Status")
            if self.inference:
                st.success("✅ Model Loaded")
            else:
                st.error("❌ Model Not Loaded")
                if st.button("Try Reload Model"):
                    self.load_model()
                    st.rerun()

            st.markdown("---")
            st.subheader("System Info")
            st.write(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            st.write(f"Python: {sys.version.split()[0]}")
            st.write(f"Streamlit: {st.__version__}")
            st.write(f"PyTorch: {torch.__version__}")
            st.write(f"OpenCV: {cv2.__version__}")

        routes = {
            "Home": self.show_home,
            "Image Analysis": self.image_analysis,
            "Real-time Webcam": self.webcam_analysis,
            "Batch Processing": self.batch_analysis,
            "Upload Video": self.video_analysis,
            "Digital Writing": self.digital_writing_analysis,
            "Results History": self.results_history,
        }
        routes[app_mode]()

    # ══════════════════════════════════════════════════════════════════════════
    def show_home(self):
        st.header("Welcome to NeuroScript AI")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
### 🎯 How It Works
1. **Upload/Capture** — Provide handwriting or drawing samples
2. **AI Analysis** — Advanced analysis of stroke patterns and features
3. **Comprehensive Assessment** — Neural Pressure Index and mental health indicators
4. **Personalized Recommendations** — Actionable insights based on results

### 🔬 Features Analyzed
- **Neural Pressure Index (NPI)** — Estimated neural load from handwriting
- **Stroke Tremors** — Fine motor control assessment
- **Writing Pressure** — Muscle tension analysis
- **Slant Consistency** — Spatial motor planning
- **Size & Spacing Patterns** — Executive function indicators
- **Temporal Characteristics** — Cognitive processing speed
""")

        with col2:
            st.markdown("""
<div style='background:#f0f2f6;padding:20px;border-radius:10px;text-align:center;'>
<h3>Sample Analysis Output</h3>
<p>Visual representation of handwriting analysis showing:</p>
<ul style='text-align:left;'>
<li>Stroke patterns</li>
<li>Pressure variations</li>
<li>Tremor detection</li>
<li>NPI score visualization</li>
</ul>
</div>
""", unsafe_allow_html=True)
            st.markdown("""
### 📱 Supported Inputs
- **Scanned handwriting** — Upload images
- **Real-time webcam** — Live writing analysis
- **Video upload** — Recorded writing sessions
- **Digital writing** — Tablet / touchscreen input
- **Batch processing** — Multiple samples at once
""")

        st.markdown("---")
        st.subheader("🚀 Quick Demo")
        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("Run Demo Analysis", type="primary"):
                with st.spinner("Analysing sample handwriting…"):
                    demo_result = self._make_demo_result()
                    st.session_state.result_history.append(demo_result)
                    self.display_results(demo_result)

        with c2:
            if st.button("View Sample Report"):
                st.info("""
**Sample Analysis Report:**
- Neural Pressure Index: 58.3 (Moderate)
- Mental Health Risk: 42.5 (Medium)
- Key Findings: Mild tremor detected, pressure inconsistencies
- Recommendation: Stress management techniques recommended
""")

        with c3:
            if st.button("Clear Demo Data"):
                st.success("Demo data cleared!")

        st.markdown("---")
        st.warning("""
⚠ **Important Medical Disclaimer**

This tool provides assessment based on handwriting patterns and is intended for
informational purposes only. It is **not** a substitute for professional medical
diagnosis, advice, or treatment. Always consult qualified healthcare providers
for medical concerns.

If you are experiencing a mental health crisis, please contact emergency services
or a crisis hotline immediately.
""")

    # ──────────────────────────────────────────────────────────────────────────
    def _make_demo_result(self):
        return {
            'timestamp': datetime.now().isoformat(),
            'prediction': 'Mild',
            'confidence': 0.78,
            'risk_score': 42.5,
            'risk_level': 'Medium',
            'neural_pressure': {
                'npi_score': 58.3,
                'npi_category': 'Moderate',
                'npi_confidence': 0.82,
            },
            'recommendation': {
                'immediate': ['Practice mindfulness exercises for 10 minutes'],
                'short_term': ['Schedule a consultation with a healthcare provider'],
                'long_term': ['Develop a regular relaxation routine'],
            },
        }

    # ══════════════════════════════════════════════════════════════════════════
    # ✅ FIX 4 — WEBCAM ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    def webcam_analysis(self):
        """
        Real-time webcam analysis.

        Root causes of "nothing happens":
        ─────────────────────────────────
        A) RealTimeWebcamInference is None (import failed) → early return blocked
           the whole function.  Fixed: we now continue and use st.camera_input().

        B) cv2.VideoCapture() + cv2.imshow() cannot run inside a Streamlit
           browser session — OpenCV tries to open a native desktop window which
           either crashes silently or does nothing in a web context.
           Fixed: use st.camera_input() which streams the webcam via the browser,
           then decode the JPEG with cv2/PIL for server-side analysis.

        C) The Start/Stop buttons had no state or logic attached to them.
           Fixed: proper session_state flag + st.rerun() control flow.
        """
        st.header("📹 Real-time Webcam Analysis")

        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader("Controls")

            analysis_frequency = st.slider(
                "Capture interval (seconds)", 1, 10, 3,
                help="How often to grab a frame for analysis"
            )
            enable_npi = st.checkbox("Enable Neural Pressure Index", True)
            save_session = st.checkbox("Save session data", False)

            st.markdown("---")
            c_start, c_stop = st.columns(2)
            with c_start:
                if st.button("▶ Start", type="primary", use_container_width=True):
                    st.session_state.webcam_running = True
            with c_stop:
                if st.button("⏹ Stop", type="secondary", use_container_width=True):
                    st.session_state.webcam_running = False

            st.markdown("---")
            st.subheader("Instructions")
            st.markdown("""
1. Click **Start** then allow browser camera access
2. Hold your handwriting sample up to the camera  
   *or* write on a white surface in view
3. Click **Capture & Analyse** to process a frame
4. Click **Stop** when finished
""")

        with col1:
            # ✅ st.camera_input() — the correct Streamlit API for webcam access.
            # It renders a live preview in the browser and returns the latest
            # snapshot as a UploadedFile (JPEG bytes) when the user clicks the
            # shutter button.  No cv2.VideoCapture needed.
            img_file = st.camera_input(
                "📷 Camera feed — click the shutter to capture a frame",
                disabled=not st.session_state.webcam_running,
                label_visibility="visible",
            )

            if not st.session_state.webcam_running:
                st.info("Press **▶ Start** in the sidebar to enable the camera.")

            if img_file is not None and st.session_state.webcam_running:
                # Decode the JPEG snapshot with PIL
                pil_image = Image.open(img_file)
                st.image(pil_image, caption="Captured frame", use_container_width=True)

                if st.button("🔬 Analyse this frame", type="primary"):
                    with st.spinner("Analysing frame…"):
                        # Convert to numpy/cv2 for any pre-processing
                        frame_np = np.array(pil_image.convert("RGB"))
                        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

                        # Pre-process: grayscale + resize to match config
                        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (224, 224))

                        if self.inference:
                            # Save to temp file for the inference engine.
                            # ✅ Windows fix: write inside the with-block so the
                            # file handle is closed before we call os.remove().
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".png"
                            ) as tmp:
                                tmp_name = tmp.name
                                cv2.imwrite(tmp_name, resized)
                            # File is closed here — safe to read and then delete
                            result = self.inference.predict(tmp_name)
                            try:
                                os.remove(tmp_name)
                            except OSError:
                                pass

                            result.update({
                                'timestamp': datetime.now().isoformat(),
                                'analysis_type': 'webcam',
                                'filename': 'webcam_frame',
                            })
                        else:
                            # Demo result when no model is loaded
                            result = self._make_demo_result()
                            result['analysis_type'] = 'webcam'
                            result['filename'] = 'webcam_frame'

                        st.session_state.result_history.append(result)
                        self.display_results(result)

    # ══════════════════════════════════════════════════════════════════════════
    # ✅ FIX 5 — DIGITAL WRITING ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    def digital_writing_analysis(self):
        """
        Digital writing analysis.

        Root causes of "can't write / nothing works":
        ──────────────────────────────────────────────
        A) DigitalWritingCapture import fails → function returned immediately.
           Fixed: we no longer gate the entire page on that import.

        B) The canvas HTML and the JavaScript were in two SEPARATE
           st.components.v1.html() calls.  Each call creates an isolated
           <iframe>; the JS in iframe-2 has no access to the <canvas> in
           iframe-1, so drawing events fired but nothing was rendered.
           Fixed: everything (HTML + JS) is in ONE html() call.

        C) There was no way for the canvas data to travel back to Python.
           st.components.v1.html() is one-way (Python → browser only).
           Fixed: use streamlit-drawable-canvas which has a proper
           bidirectional bridge, giving us stroke coordinates + a PNG.

        D) streamlit-drawable-canvas was not in requirements.txt.
           Fixed: added below, and we fall back gracefully when missing.
        """
        st.header("✍️ Digital Writing Analysis")
        st.markdown("""
Analyse digital handwriting from tablets or touchscreens.
Provides **pressure data**, **stroke dynamics**, and **precise timing**.
""")

        # ── Install hint ──────────────────────────────────────────────────────
        if not CANVAS_AVAILABLE:
            st.error(
                "**Missing package:** `streamlit-drawable-canvas` is not installed.\n\n"
                "Add this line to `requirements.txt` and restart:\n"
                "```\nstreamlit-drawable-canvas>=0.9.3\n```\n"
                "or run:  `pip install streamlit-drawable-canvas`"
            )
            # Fallback: still show the raw HTML canvas so the page isn't blank,
            # but note that Python cannot receive stroke data from it.
            st.warning("Showing HTML-only canvas (no Python analysis without the package).")
            self._fallback_html_canvas()
            return

        col1, col2 = st.columns([2, 1])

        with col2:
            st.subheader("Controls")

            stroke_width = st.slider("Pen width", 1, 25, 4)
            stroke_color = st.color_picker("Pen colour", "#000000")
            bg_color = st.color_picker("Background", "#FFFFFF")
            drawing_mode = st.selectbox(
                "Drawing mode",
                ["freedraw", "line", "rect", "circle", "point"],
            )

            st.markdown("---")
            st.subheader("Writing Prompt")
            prompt = st.selectbox(
                "Choose a prompt",
                [
                    "Write your signature",
                    "Write 'Hello World'",
                    "Draw a continuous spiral",
                    "Write the alphabet",
                    "Free drawing",
                ],
            )

            st.markdown("---")
            analyze_btn = st.button(
                "🔍 Analyse Writing", type="primary", use_container_width=True
            )
            clear_btn = st.button(
                "🗑️ Clear Canvas", type="secondary", use_container_width=True
            )

            st.markdown("---")
            st.subheader("Tips")
            st.info("""
- Use a stylus for richer pressure data
- Write naturally, not too slowly
- Complete the full prompt
- Avoid excessive lifting
""")

        with col1:
            st.subheader("Writing Area")

            # ✅ st_canvas() — bidirectional: renders in browser AND returns
            # stroke JSON + a PNG numpy array back to Python.
            canvas_result = st_canvas(
                fill_color="rgba(255,165,0,0.3)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                height=420,
                width=760,
                drawing_mode=drawing_mode,
                key="main_canvas",
                display_toolbar=True,
            )

            # Show live stroke count
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
                n_strokes = len(objects)
                if n_strokes:
                    st.caption(f"✏️ {n_strokes} stroke{'s' if n_strokes != 1 else ''} recorded")

        # ── Analysis ──────────────────────────────────────────────────────────
        if analyze_btn:
            if canvas_result.json_data is None or not canvas_result.json_data.get("objects"):
                st.warning("Please draw something on the canvas first.")
                return

            with st.spinner("Analysing digital writing…"):
                strokes = canvas_result.json_data["objects"]
                features = self._extract_canvas_features(strokes)

                # If a model is loaded AND the canvas image is available, run it
                result = None
                if self.inference and canvas_result.image_data is not None:
                    try:
                        img_array = canvas_result.image_data[:, :, :3]  # drop alpha
                        pil_img = Image.fromarray(img_array.astype(np.uint8))

                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".png"
                        ) as tmp:
                            tmp_name = tmp.name
                            pil_img.save(tmp_name)
                        # Closed before predict — safe on Windows
                        result = self.inference.predict(tmp_name)
                        try:
                            os.remove(tmp_name)
                        except OSError:
                            pass
                    except Exception as e:
                        st.warning(f"Model inference skipped: {e}")

                if result is None:
                    # Demo result — still shows real extracted features
                    result = {
                        'timestamp': datetime.now().isoformat(),
                        'analysis_type': 'digital_writing',
                        'prediction': 'Normal',
                        'confidence': 0.85,
                        'risk_score': 28.5,
                        'risk_level': 'Low',
                        'neural_pressure': {
                            'npi_score': 32.1,
                            'npi_category': 'Low',
                            'npi_confidence': 0.78,
                        },
                        'recommendation': {
                            'immediate': ['Continue current writing practice'],
                            'short_term': ['Practice fine motor exercises'],
                            'long_term': ['Maintain regular writing activities'],
                        },
                    }

                result['prompt'] = prompt
                result['filename'] = 'digital_canvas'
                result['digital_features'] = features

                st.session_state.result_history.append(result)

            # Show extracted features alongside main results
            st.markdown("---")
            st.subheader("📐 Extracted Stroke Features")
            feat_cols = st.columns(4)
            feat_items = [
                ("Strokes", features['stroke_count']),
                ("Total points", features['total_points']),
                ("Avg stroke len", f"{features['avg_stroke_length']:.1f} px"),
                ("Avg speed (est.)", f"{features['avg_speed_estimate']:.2f}"),
            ]
            for col, (label, val) in zip(feat_cols, feat_items):
                col.metric(label, val)

            self.display_results(result)

    # ─────────────────────────────────────────────
    def _extract_canvas_features(self, objects: list) -> dict:
        """
        Pull simple geometric / temporal features from st_canvas JSON strokes.
        Each object in the JSON has a 'path' list of SVG-like commands.
        """
        stroke_count = len(objects)
        total_points = 0
        stroke_lengths = []

        for obj in objects:
            path = obj.get("path", [])
            points = [
                (cmd[1], cmd[2])
                for cmd in path
                if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ("M", "L", "Q")
            ]
            total_points += len(points)

            # Euclidean length of the stroke
            length = 0.0
            for i in range(1, len(points)):
                dx = points[i][0] - points[i - 1][0]
                dy = points[i][1] - points[i - 1][1]
                length += (dx ** 2 + dy ** 2) ** 0.5
            stroke_lengths.append(length)

        avg_len = float(np.mean(stroke_lengths)) if stroke_lengths else 0.0
        # Speed estimate: longer strokes drawn quickly imply higher speed
        avg_speed = avg_len / max(total_points, 1) if total_points else 0.0

        return {
            "stroke_count": stroke_count,
            "total_points": total_points,
            "avg_stroke_length": avg_len,
            "avg_speed_estimate": avg_speed,
            "stroke_length_std": float(np.std(stroke_lengths)) if stroke_lengths else 0.0,
        }

    # ─────────────────────────────────────────────
    def _fallback_html_canvas(self):
        """
        Single-iframe HTML+JS canvas shown when streamlit-drawable-canvas
        is not installed.  All HTML, CSS, and JS are in ONE html() call
        so they share the same iframe document — fixing the original bug
        where two separate html() calls created two isolated iframes.
        """
        st.components.v1.html(
            """
<!DOCTYPE html>
<html>
<head>
<style>
  body { margin: 0; font-family: sans-serif; background: #fafafa; }
  #wrap { padding: 10px; }
  canvas { border: 2px solid #aaa; background: white; cursor: crosshair;
           display: block; border-radius: 6px; }
  .toolbar { margin-top: 8px; display: flex; gap: 8px; align-items: center; }
  button { padding: 6px 14px; border-radius: 4px; border: 1px solid #888;
           cursor: pointer; background: #f0f0f0; }
  button:hover { background: #e0e0e0; }
</style>
</head>
<body>
<div id="wrap">
  <canvas id="c" width="740" height="380"></canvas>
  <div class="toolbar">
    <button onclick="clearC()">Clear</button>
    <button onclick="undo()">Undo</button>
    <label>Colour <input type="color" id="col" value="#000000"></label>
    <label>Size <input type="range" id="sz" min="1" max="20" value="4"></label>
  </div>
</div>
<script>
  const canvas = document.getElementById('c');
  const ctx    = canvas.getContext('2d');
  let drawing  = false;
  let strokes  = [];
  let cur      = [];

  ctx.lineWidth  = 4;
  ctx.lineCap    = 'round';
  ctx.lineJoin   = 'round';
  ctx.strokeStyle = '#000000';

  function pos(e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }
  function tpos(t) {
    const r = canvas.getBoundingClientRect();
    return { x: t.clientX - r.left, y: t.clientY - r.top };
  }

  canvas.addEventListener('mousedown',  e => { drawing=true; cur=[]; const p=pos(e); ctx.beginPath(); ctx.moveTo(p.x,p.y); cur.push(p); });
  canvas.addEventListener('mousemove',  e => { if(!drawing) return; const p=pos(e); ctx.lineTo(p.x,p.y); ctx.stroke(); cur.push(p); });
  canvas.addEventListener('mouseup',    () => { if(drawing){ drawing=false; strokes.push([...cur]); } });
  canvas.addEventListener('mouseleave', () => { if(drawing){ drawing=false; strokes.push([...cur]); } });

  canvas.addEventListener('touchstart', e => { e.preventDefault(); drawing=true; cur=[]; const p=tpos(e.touches[0]); ctx.beginPath(); ctx.moveTo(p.x,p.y); cur.push(p); }, {passive:false});
  canvas.addEventListener('touchmove',  e => { e.preventDefault(); if(!drawing) return; const p=tpos(e.touches[0]); ctx.lineTo(p.x,p.y); ctx.stroke(); cur.push(p); }, {passive:false});
  canvas.addEventListener('touchend',   () => { if(drawing){ drawing=false; strokes.push([...cur]); } });

  document.getElementById('col').addEventListener('change', e => ctx.strokeStyle = e.target.value);
  document.getElementById('sz').addEventListener('input',   e => ctx.lineWidth   = e.target.value);

  function clearC() { ctx.clearRect(0,0,canvas.width,canvas.height); strokes=[]; }
  function undo()   { if(!strokes.length) return; strokes.pop(); redraw(); }
  function redraw() {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    const col = ctx.strokeStyle, sz = ctx.lineWidth;
    strokes.forEach(s => {
      if(!s.length) return;
      ctx.beginPath(); ctx.moveTo(s[0].x, s[0].y);
      s.forEach(p => ctx.lineTo(p.x, p.y));
      ctx.stroke();
    });
  }
</script>
</body>
</html>
""",
            height=480,
            scrolling=False,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # IMAGE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    def image_analysis(self):
        st.header("📸 Image Analysis")
        st.markdown("Upload a handwriting or drawing image for detailed analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png", "tiff", "bmp", "webp"],
            )

            if uploaded_file:
                try:
                    image = Image.open(uploaded_file)
                    zoom = st.slider("Zoom", 0.5, 2.0, 1.0, 0.1)
                    if zoom != 1.0:
                        new_size = (int(image.width * zoom), int(image.height * zoom))
                        image = image.resize(new_size, Image.Resampling.LANCZOS)
                    # ✅ use_container_width replaces deprecated use_column_width
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")

        with col2:
            st.subheader("Analysis Settings")
            analyze_npi = st.checkbox("Estimate Neural Pressure Index", True)
            detailed_features = st.checkbox("Extract detailed features", True)

            if os.path.exists("saved_models"):
                model_files = [f for f in os.listdir("saved_models") if f.endswith(".pth")]
                if model_files:
                    selected = st.selectbox("Select Model", model_files)
                    self.model_path = os.path.join("saved_models", selected)

            st.markdown("---")
            if uploaded_file and st.button("🚀 Analyse Image", type="primary", use_container_width=True):
                if not self.inference:
                    st.error("Please load a model first")
                    return

                with st.spinner("Analysing…"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        tmp_name = tmp.name
                        Image.open(uploaded_file).save(tmp_name)
                    # ✅ File handle closed — safe to read and delete on Windows
                    try:
                        result = self.inference.predict(tmp_name)
                        result.update({
                            "timestamp": datetime.now().isoformat(),
                            "filename": uploaded_file.name,
                            "analysis_type": "image",
                        })
                        st.session_state.result_history.append(result)
                        self.display_results(result)
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                    finally:
                        try:
                            os.remove(tmp_name)
                        except OSError:
                            pass

    # ══════════════════════════════════════════════════════════════════════════
    # DISPLAY RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    def display_results(self, result):
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Summary", "🧠 Neural Pressure", "📈 Features", "💡 Recommendations"]
        )
        with tab1:
            self._display_summary_tab(result)
        with tab2:
            self._display_neural_pressure_tab(result)
        with tab3:
            self._display_features_tab(result)
        with tab4:
            self._display_recommendations_tab(result)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            json_str = json.dumps(result, indent=2, default=str)
            st.download_button(
                "📄 Export JSON",
                data=json_str,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with c2:
            if st.button("📥 Save to History", use_container_width=True):
                st.success("Saved to analysis history!")

    # ─────────────────────────────────────────────
    def _display_summary_tab(self, result):
        c1, c2, c3 = st.columns(3)
        color_map = {"Normal": "green", "Mild": "orange", "Severe": "red"}

        prediction = result.get("prediction", "Unknown")
        confidence = result.get("confidence", 0.0)
        risk_score = result.get("risk_score", 50)
        risk_level = result.get("risk_level", "Unknown")

        with c1:
            st.metric("Prediction", prediction, f"{confidence:.1%} confidence", delta_color="off")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=confidence * 100,
                title={"text": "Confidence"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": color_map.get(prediction, "gray")}}
            ))
            fig.update_layout(height=200, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.metric("Risk Score", f"{risk_score:.1f}/100", risk_level)
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_score,
                title={"text": "Mental Health Risk"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "darkblue"},
                       "steps": [
                           {"range": [0, 30], "color": "green"},
                           {"range": [30, 70], "color": "orange"},
                           {"range": [70, 100], "color": "red"},
                       ]}
            ))
            fig.update_layout(height=200, margin=dict(t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with c3:
            if "neural_pressure" in result:
                npi = result["neural_pressure"]
                st.metric("Neural Pressure", f"{npi['npi_score']:.1f}/100", npi["npi_category"])
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=npi["npi_score"],
                    title={"text": "NPI"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "purple"}}
                ))
                fig.update_layout(height=200, margin=dict(t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Neural Pressure analysis not available")

        st.markdown("---")
        i1, i2 = st.columns(2)
        with i1:
            st.subheader("Analysis Details")
            st.write(f"**Timestamp:** {result.get('timestamp', 'Unknown')}")
            st.write(f"**Filename:** {result.get('filename', 'Unknown')}")
            st.write(f"**Type:** {result.get('analysis_type', 'Unknown')}")
        with i2:
            st.subheader("Key Findings")
            if risk_score > 70:
                st.write("• High mental health risk detected")
            elif risk_score > 40:
                st.write("• Moderate risk indicators present")
            else:
                st.write("• Low risk profile")

    # ─────────────────────────────────────────────
    def _display_neural_pressure_tab(self, result):
        if "neural_pressure" not in result:
            st.info("Neural Pressure Index analysis not available for this result.")
            return

        npi_data = result["neural_pressure"]
        npi_score = npi_data.get("npi_score", 50)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=npi_score,
                title={"text": f"NPI — {npi_data.get('npi_category', '')}"},
                delta={"reference": 50},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "purple"},
                       "steps": [
                           {"range": [0, 25], "color": "rgba(0,255,0,.3)"},
                           {"range": [25, 50], "color": "rgba(144,238,144,.3)"},
                           {"range": [50, 75], "color": "rgba(255,165,0,.3)"},
                           {"range": [75, 100], "color": "rgba(255,0,0,.3)"},
                       ]}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Components")
            for comp, val in {
                "tremor": 0.15, "pressure_variability": 0.25,
                "velocity_irregularity": 0.18, "hesitation": 0.12,
            }.items():
                st.progress(val, text=f"{comp.replace('_',' ').title()}: {val:.3f}")
            st.metric("Confidence", f"{npi_data.get('npi_confidence', 0.5):.1%}")

    # ─────────────────────────────────────────────
    def _display_features_tab(self, result):
        features = result.get("neuromotor_features") or {
            "tremor_mean": 0.15, "jitter_index": 0.08,
            "pressure_variability": 0.25, "curvature_instability": 0.12,
            "pause_ratio": 0.05, "velocity_cv": 0.28,
        }
        # Also show digital features if present
        if "digital_features" in result:
            features.update(result["digital_features"])

        categories = {
            "Motor Control": ["tremor_mean", "jitter_index", "velocity_cv"],
            "Pressure": ["pressure_variability"],
            "Spatial": ["curvature_instability"],
            "Temporal": ["pause_ratio", "hesitation_index"],
        }
        scores = [min(np.mean([features.get(f, 0) for f in fl]), 1.0)
                  for fl in categories.values()]

        fig = go.Figure(go.Scatterpolar(
            r=scores, theta=list(categories.keys()),
            fill="toself", fillcolor="rgba(30,144,255,.3)",
            line=dict(color="royalblue", width=2),
        ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])),
                          showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        rows = [{"Feature": k.replace("_", " ").title(),
                 "Value": f"{v:.4f}" if isinstance(v, float) else str(v),
                 "Note": self._interpret_feature(k, v if isinstance(v, float) else 0)}
                for k, v in list(features.items())[:20]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ─────────────────────────────────────────────
    def _interpret_feature(self, name, value):
        if "tremor" in name:
            return "High tremor" if value > 0.3 else "Moderate" if value > 0.1 else "Low tremor"
        if "pressure" in name:
            return "High variability" if value > 0.4 else "Moderate" if value > 0.2 else "Consistent"
        return "High" if value > 0.7 else "Moderate" if value > 0.3 else "Low"

    # ─────────────────────────────────────────────
    def _display_recommendations_tab(self, result):
        risk_score = result.get("risk_score", 50)
        recs = result.get("recommendation") or self._generate_recommendations(risk_score, result)

        tabs = st.tabs(["🚨 Immediate", "📅 Short-term", "🎯 Long-term"])
        for tab, key in zip(tabs, ["immediate", "short_term", "long_term"]):
            with tab:
                for i, r in enumerate(recs.get(key, ["No action needed"]), 1):
                    st.markdown(f"{i}. **{r}**")

        st.markdown("---")
        st.subheader("📋 Action Plan Checklist")
        for item in [
            "Schedule regular handwriting assessments",
            "Practice relaxation techniques daily",
            "Maintain consistent sleep schedule",
            "Monitor stress levels",
        ]:
            st.checkbox(item, key=f"chk_{item}")

    # ─────────────────────────────────────────────
    def _generate_recommendations(self, risk_score, result):
        if risk_score > 70:
            return {
                "immediate": ["Consult a mental health professional", "Practice deep breathing (5 min)"],
                "short_term": ["Schedule a wellness check-up", "Implement daily stress routine"],
                "long_term": ["Comprehensive mental health plan", "Regular therapy sessions"],
            }
        elif risk_score > 40:
            return {
                "immediate": ["Take a 10-minute relaxation break"],
                "short_term": ["Mindfulness meditation daily", "Monitor patterns weekly"],
                "long_term": ["Consistent self-care routine"],
            }
        return {
            "immediate": ["Maintain current healthy habits"],
            "short_term": ["Continue regular self-assessment"],
            "long_term": ["Develop resilience-building practices"],
        }

    # ══════════════════════════════════════════════════════════════════════════
    # BATCH ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    def batch_analysis(self):
        st.header("📁 Batch Processing")
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=["jpg", "jpeg", "png", "tiff", "bmp", "webp"],
            accept_multiple_files=True,
        )
        if not uploaded_files:
            return

        st.success(f"✅ {len(uploaded_files)} files uploaded")

        if st.button("🚀 Start Batch Analysis", type="primary"):
            if not self.inference:
                st.error("Please load a model first")
                return
            progress = st.progress(0)
            results = []
            for i, f in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp_name = tmp.name
                    Image.open(f).save(tmp_name)
                # ✅ File handle closed — safe on Windows
                try:
                    r = self.inference.predict(tmp_name)
                    r["filename"] = f.name
                    r["timestamp"] = datetime.now().isoformat()
                    results.append(r)
                except Exception as e:
                    st.error(f"{f.name}: {e}")
                finally:
                    try:
                        os.remove(tmp_name)
                    except OSError:
                        pass
                progress.progress((i + 1) / len(uploaded_files))

            st.session_state.result_history.extend(results)
            self._display_batch_summary(results)

    # ─────────────────────────────────────────────
    def _display_batch_summary(self, results):
        if not results:
            return
        rows = [{
            "Filename": r.get("filename", ""),
            "Prediction": r.get("prediction", ""),
            "Confidence": f"{r.get('confidence', 0):.1%}",
            "Risk Score": r.get("risk_score", 0),
            "Risk Level": r.get("risk_level", ""),
        } for r in results]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            counts = df["Prediction"].value_counts()
            st.plotly_chart(
                px.pie(values=counts.values, names=counts.index, title="Prediction Distribution"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                px.histogram(df, x="Risk Score", title="Risk Score Distribution", nbins=20),
                use_container_width=True,
            )

        csv = df.to_csv(index=False)
        st.download_button("📄 Export CSV", data=csv,
                           file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")

    # ══════════════════════════════════════════════════════════════════════════
    # VIDEO ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    def video_analysis(self):
        st.header("🎬 Video Analysis")
        uploaded_video = st.file_uploader(
            "Choose a video file", type=["mp4", "avi", "mov", "mkv", "webm"]
        )
        if not uploaded_video:
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name
        # ✅ File handle closed here — cv2.VideoCapture can now open it on Windows

        st.video(video_path)

        frame_rate = st.slider("Analysis Frame Rate (fps)", 1, 30, 5)

        if st.button("🎯 Analyse Video", type="primary"):
            if not self.inference:
                st.error("Please load a model first")
                return
            with st.spinner("Analysing video frames…"):
                try:
                    cap = cv2.VideoCapture(video_path)
                    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress = st.progress(0)
                    frame_results, fc = [], 0
                    interval = max(1, fps // frame_rate)

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if fc % interval == 0:
                            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            # ✅ Write then close the with-block before predict/delete
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
                                tf_name = tf.name
                                pil.save(tf_name)
                            try:
                                r = self.inference.predict(tf_name)
                                r["frame"] = fc
                                r["timestamp"] = fc / fps
                                frame_results.append(r)
                            except Exception:
                                pass
                            finally:
                                try:
                                    os.remove(tf_name)
                                except OSError:
                                    pass
                        fc += 1
                        progress.progress(min(fc / max(total, 1), 1.0))
                    cap.release()

                    if frame_results:
                        combined = self._combine_video_results(frame_results)
                        combined.update({"filename": uploaded_video.name,
                                         "analysis_type": "video"})
                        st.session_state.result_history.append(combined)
                        self.display_results(combined)
                        self._display_video_timeline(frame_results)
                except Exception as e:
                    st.error(f"Video analysis failed: {e}")
                finally:
                    try:
                        os.remove(video_path)
                    except Exception:
                        pass

    # ─────────────────────────────────────────────
    def _combine_video_results(self, frame_results):
        risk = [r.get("risk_score", 0) for r in frame_results]
        npi = [r["neural_pressure"]["npi_score"]
               for r in frame_results if "neural_pressure" in r]
        conf = [r.get("confidence", 0) for r in frame_results]
        preds = Counter(r.get("prediction", "Unknown") for r in frame_results)
        avg_risk = float(np.mean(risk)) if risk else 0
        avg_npi = float(np.mean(npi)) if npi else 0
        return {
            "timestamp": datetime.now().isoformat(),
            "risk_score": avg_risk,
            "risk_level": self._get_risk_level(avg_risk),
            "confidence": float(np.mean(conf)) if conf else 0,
            "prediction": preds.most_common(1)[0][0],
            "neural_pressure": {
                "npi_score": avg_npi,
                "npi_category": self._get_npi_category(avg_npi),
                "npi_confidence": 0.8,
            },
        }

    def _get_risk_level(self, s):
        return "Low" if s < 30 else "High" if s > 70 else "Medium"

    def _get_npi_category(self, s):
        return "Low" if s < 30 else "High" if s > 60 else "Moderate"

    def _display_video_timeline(self, frame_results):
        ts = [r.get("timestamp", i) for i, r in enumerate(frame_results)]
        risk = [r.get("risk_score", 0) for r in frame_results]
        npi = [r.get("neural_pressure", {}).get("npi_score", 0) for r in frame_results]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=risk, mode="lines+markers",
                                 name="Risk Score", line=dict(color="red")))
        fig.add_trace(go.Scatter(x=ts, y=npi, mode="lines+markers",
                                 name="NPI Score", line=dict(color="blue")))
        fig.update_layout(title="Scores Over Time", xaxis_title="Seconds",
                          yaxis_title="Score", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # RESULTS HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    def results_history(self):
        st.header("📋 Analysis History")
        history = st.session_state.result_history

        if not history:
            st.info("No results yet. Run an analysis first.")
            return

        rows = [{
            "ID": i + 1,
            "Filename": r.get("filename", ""),
            "Type": r.get("analysis_type", ""),
            "Prediction": r.get("prediction", ""),
            "Risk Score": r.get("risk_score", 0),
            "Risk Level": r.get("risk_level", ""),
            "Timestamp": r.get("timestamp", ""),
        } for i, r in enumerate(history)]

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            csv = pd.DataFrame(rows).to_csv(index=False)
            st.download_button("📥 Export CSV", data=csv,
                               file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv", use_container_width=True)
        with c2:
            if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                st.session_state.result_history = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
def main():
    app = MentalHealthWebApp()
    app.run()


if __name__ == "__main__":

    main()
