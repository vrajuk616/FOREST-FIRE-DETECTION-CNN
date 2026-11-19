import os
import io
import json
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from matplotlib import cm

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

# ----------------------------------------------------------------------------
# Config (matches the notebook: 150x150 RGB, rescale 1/255, sigmoid output)
# class_indices = {'fire': 0, 'nofire': 1}  ->  sigmoid output = P(nofire)
# ----------------------------------------------------------------------------
MODEL_PATH = "FFD.keras"
IMG_SIZE = (150, 150)

st.set_page_config(page_title="Forest Fire Detection", page_icon="🔥", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(135deg,#0f2027 0%,#1b2b22 45%,#2c1810 100%);}
    h1,h2,h3,h4,p,label,span,div {color:#f5efe6;}
    .hero {padding:28px 32px;border-radius:20px;margin-bottom:18px;
           background:linear-gradient(120deg,#ff512f 0%,#f09819 55%,#2e7d32 130%);
           box-shadow:0 10px 30px rgba(255,81,47,.35);}
    .hero h1 {margin:0;font-size:2.4rem;color:#fff;}
    .hero p {margin:6px 0 0;color:#fff3e0;font-size:1.05rem;}
    .card {padding:18px 22px;border-radius:16px;background:rgba(255,255,255,.07);
           border:1px solid rgba(255,255,255,.12);backdrop-filter:blur(6px);}
    .fire {background:linear-gradient(120deg,#d32f2f,#ff6f00);padding:18px;border-radius:16px;
           text-align:center;font-size:1.6rem;font-weight:800;color:#fff;
           box-shadow:0 0 25px rgba(255,87,34,.6);}
    .safe {background:linear-gradient(120deg,#1b5e20,#43a047);padding:18px;border-radius:16px;
           text-align:center;font-size:1.6rem;font-weight:800;color:#fff;
           box-shadow:0 0 25px rgba(67,160,71,.5);}
    .small {font-size:.85rem;opacity:.8;}
    [data-testid="stSidebar"] {background:rgba(0,0,0,.35);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🔥 Forest Fire Detection using Deep Learning</h1>
      <p>Upload a forest / landscape image. A CNN classifies it as <b>Fire</b> or <b>No Fire</b>
      and a Grad-CAM heatmap shows where the model looked.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def build_model():
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.Input(shape=(150, 150, 3)),
        L.Conv2D(32, (3, 3), activation="relu"), L.MaxPooling2D((2, 2)),
        L.Conv2D(64, (3, 3), activation="relu"), L.MaxPooling2D((2, 2)),
        L.Conv2D(128, (3, 3), activation="relu"), L.MaxPooling2D((2, 2)),
        L.Flatten(), L.Dense(512, activation="relu"), L.Dropout(0.5),
        L.Dense(1, activation="sigmoid"),
    ])


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    # Option 1: split float16 weights (manifest.json + weights_*.bin) next to app.py
    if os.path.exists("manifest.json"):
        man = json.load(open("manifest.json"))
        blob = b"".join(open(f, "rb").read() for f in man["files"])
        flat = np.frombuffer(blob, dtype="<f2").astype("float32")
        m, pos, ws = build_model(), 0, []
        for shp in man["shapes"]:
            n = int(np.prod(shp))
            ws.append(flat[pos:pos + n].reshape(shp))
            pos += n
        m.set_weights(ws)
        return m
    # Option 2: full FFD.keras file (optionally downloaded from MODEL_URL)
    if not os.path.exists(MODEL_PATH):
        url = os.environ.get("MODEL_URL") or st.secrets.get("MODEL_URL", None)
        if url:
            urllib.request.urlretrieve(url, MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    return None


model = load_model()
if model is None:
    st.error(
        "Model files not found. Upload **manifest.json**, **weights_0.bin** and **weights_1.bin** "
        "(created by export_weights.py) to the same GitHub repo as app.py."
    )
    st.stop()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, 0)


def predict(arr: np.ndarray) -> tuple[float, float]:
    p_nofire = float(model.predict(arr, verbose=0)[0][0])
    return 1.0 - p_nofire, p_nofire  # (P(fire), P(nofire))


def grad_cam(arr: np.ndarray) -> np.ndarray:
    """Grad-CAM for the 'fire' class (= 1 - sigmoid output)."""
    convs = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    last_conv = convs[-1]
    x = tf.convert_to_tensor(arr)
    with tf.GradientTape() as tape:
        h = x
        conv_out = None
        for layer in model.layers:
            h = layer(h)
            if layer is last_conv:
                conv_out = h
        fire_score = 1.0 - h[:, 0]
    grads = tape.gradient(fire_score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    img = img.convert("RGB")
    heat = Image.fromarray(np.uint8(255 * cam)).resize(img.size, Image.BICUBIC)
    heat = np.asarray(heat) / 255.0
    colored = (cm.jet(heat)[..., :3] * 255).astype("uint8")
    return Image.fromarray(
        np.uint8((1 - alpha) * np.asarray(img) + alpha * colored)
    )


def gauge(p_fire: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_fire * 100,
            number={"suffix": "%", "font": {"size": 44, "color": "#fff"}},
            title={"text": "Fire Probability", "font": {"size": 18, "color": "#fff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#ddd"},
                "bar": {"color": "#ff3d00"},
                "bgcolor": "rgba(255,255,255,0.08)",
                "steps": [
                    {"range": [0, 30], "color": "#2e7d32"},
                    {"range": [30, 60], "color": "#f9a825"},
                    {"range": [60, 100], "color": "#b71c1c"},
                ],
            },
        )
    )
    fig.update_layout(height=290, margin=dict(t=60, b=10, l=30, r=30),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#fff"})
    return fig


def prob_bar(p_fire: float, p_nofire: float) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=["Fire", "No Fire"],
            y=[p_fire * 100, p_nofire * 100],
            marker_color=["#ff5722", "#43a047"],
            text=[f"{p_fire*100:.1f}%", f"{p_nofire*100:.1f}%"],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=290, yaxis=dict(range=[0, 115], title="Probability (%)"),
        margin=dict(t=30, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#fff"},
    )
    return fig


def rgb_hist(img: Image.Image) -> go.Figure:
    a = np.asarray(img.convert("RGB").resize(IMG_SIZE))
    fig = go.Figure()
    for i, (name, col) in enumerate([("Red", "#ff5252"), ("Green", "#69f0ae"), ("Blue", "#40c4ff")]):
        h, _ = np.histogram(a[..., i], bins=32, range=(0, 255))
        fig.add_trace(go.Scatter(x=np.linspace(0, 255, 32), y=h, name=name,
                                 fill="tozeroy", line=dict(color=col), opacity=0.6))
    fig.update_layout(height=290, title="Colour channel distribution",
                      margin=dict(t=50, b=10, l=10, r=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#fff"})
    return fig


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Fire alert threshold", 0.10, 0.90, 0.50, 0.05,
                          help="Image is flagged as FIRE if P(fire) >= threshold")
    show_cam = st.checkbox("Show Grad-CAM heatmap", True)
    alpha = st.slider("Heatmap intensity", 0.2, 0.8, 0.45, 0.05, disabled=not show_cam)
    st.markdown("---")
    st.subheader("🧠 Model")
    st.markdown(
        """
        **CNN (Keras / TensorFlow)**
        - Input: 150 × 150 × 3
        - Conv32 → Conv64 → Conv128 (+MaxPool)
        - Dense 512 + Dropout 0.5
        - Sigmoid output, Adam, binary cross-entropy
        - Dataset: The Wildfire Dataset (Kaggle)
        """
    )
    st.caption("For demonstration; not a substitute for official fire-warning systems.")

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
files = st.file_uploader(
    "📤 Upload image(s)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
)

if not files:
    st.markdown(
        '<div class="card">👆 Upload one or more images to get started. '
        'Try aerial forest shots, smoke plumes, or normal landscapes.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

rows = []
for f in files:
    img = Image.open(io.BytesIO(f.getvalue()))
    arr = preprocess(img)
    p_fire, p_nofire = predict(arr)
    is_fire = p_fire >= threshold
    rows.append({"Image": f.name, "P(Fire) %": round(p_fire * 100, 2),
                 "P(No Fire) %": round(p_nofire * 100, 2),
                 "Verdict": "🔥 FIRE" if is_fire else "🌲 NO FIRE"})

    st.markdown(f"### 🖼️ {f.name}")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(img.convert("RGB"), caption="Original", use_container_width=True)
    with c2:
        if show_cam:
            cam = grad_cam(arr)
            st.image(overlay(img, cam, alpha), caption="Grad-CAM (regions driving the 'fire' score)",
                     use_container_width=True)
        else:
            st.image(img.convert("RGB"), use_container_width=True)

    if is_fire:
        st.markdown(f'<div class="fire">🔥 FIRE DETECTED — {p_fire*100:.1f}% confidence</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="safe">🌲 NO FIRE — {p_nofire*100:.1f}% confidence</div>',
                    unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(gauge(p_fire), use_container_width=True, key=f"g_{f.name}")
    g2.plotly_chart(prob_bar(p_fire, p_nofire), use_container_width=True, key=f"b_{f.name}")
    g3.plotly_chart(rgb_hist(img), use_container_width=True, key=f"h_{f.name}")
    st.markdown("---")

if len(rows) > 1:
    df = pd.DataFrame(rows)
    st.subheader("📊 Batch summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Images", len(df))
    m2.metric("Fire detected", int((df["Verdict"] == "🔥 FIRE").sum()))
    m3.metric("No fire", int((df["Verdict"] == "🌲 NO FIRE").sum()))
    fig = go.Figure(go.Bar(x=df["Image"], y=df["P(Fire) %"],
                           marker=dict(color=df["P(Fire) %"], colorscale="YlOrRd", cmin=0, cmax=100)))
    fig.update_layout(height=320, yaxis_title="P(Fire) %", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={"color": "#fff"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Download results (CSV)", df.to_csv(index=False).encode(),
                       "fire_predictions.csv", "text/csv")
