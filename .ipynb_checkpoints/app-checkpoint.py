import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Document Classifier",
    page_icon="🪪",
    layout="centered"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #f0f4f8 !important;
    font-family: 'DM Sans', sans-serif;
    color: #1a202c;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 780px !important; }

.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a56db 50%, #1e40af 100%);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 60% 40%, rgba(255,255,255,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: #ffffff;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.15;
    margin-bottom: 0.6rem;
    letter-spacing: -0.02em;
}
.hero h1 span { color: #bfdbfe; }
.hero p {
    color: rgba(255,255,255,0.75);
    font-size: 0.95rem;
    font-weight: 300;
    max-width: 440px;
    margin: 0 auto;
    line-height: 1.6;
}

.stats-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.stat-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
    transition: border-color 0.2s, box-shadow 0.2s;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.stat-card:hover { border-color: #3b82f6; box-shadow: 0 4px 16px rgba(59,130,246,0.12); }
.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #1d4ed8;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.stat-label {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

.upload-section {
    background: #ffffff;
    border: 1.5px dashed #93c5fd;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    transition: border-color 0.2s, background 0.2s;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.upload-section:hover { border-color: #3b82f6; background: #eff6ff; }
.upload-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.upload-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #1e3a5f;
    margin-bottom: 0.3rem;
}
.upload-sub { font-size: 0.82rem; color: #94a3b8; }

[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.35) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(59,130,246,0.45) !important;
}
div.stButton > button:active { transform: translateY(0px) !important; }

.result-card {
    background: #ffffff;
    border: 1px solid #bfdbfe;
    border-radius: 18px;
    padding: 2rem;
    margin-top: 1.5rem;
    animation: fadeUp 0.4s ease;
    box-shadow: 0 4px 20px rgba(59,130,246,0.1);
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label {
    font-size: 0.72rem;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 500;
    margin-bottom: 0.4rem;
}
.result-class {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #1e3a5f;
    margin-bottom: 0.3rem;
}
.result-confidence {
    font-size: 0.9rem;
    color: #1d4ed8;
    font-weight: 500;
    margin-bottom: 1.5rem;
}

.prob-row { margin-bottom: 0.85rem; }
.prob-header { display: flex; justify-content: space-between; margin-bottom: 0.3rem; }
.prob-name { font-size: 0.82rem; color: #64748b; font-weight: 400; }
.prob-value { font-size: 0.82rem; color: #1d4ed8; font-weight: 600; }
.prob-track {
    background: #e2e8f0;
    border-radius: 100px;
    height: 6px;
    overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 100px; transition: width 0.6s ease; }
.prob-fill-high   { background: linear-gradient(90deg, #1d4ed8, #60a5fa); }
.prob-fill-medium { background: linear-gradient(90deg, #94a3b8, #cbd5e1); }
.prob-fill-low    { background: #e2e8f0; }

.divider { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.75rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #e2e8f0;
}
.footer span { color: #64748b; }
[data-testid="caption"] { color: #94a3b8 !important; font-size: 0.78rem !important; }
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL DEFINITION
# ─────────────────────────────────────────────
class CNNNet(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CNNNet(num_classes=3).to(device)
    model.load_state_dict(torch.load("output/model.pt", map_location=device))
    model.eval()
    return model, device


# ─────────────────────────────────────────────
#  TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

CLASS_NAMES = ["Driving License", "Others", "Social Security"]
CLASS_ICONS = ["🪪", "📄", "🔐"]


# ─────────────────────────────────────────────
#  PREDICT FUNCTION
# ─────────────────────────────────────────────
def predict(image: Image.Image, model, device):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output        = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
        predicted_idx = int(np.argmax(probabilities))
    return predicted_idx, probabilities


# ─────────────────────────────────────────────
#  RENDER RESULT  (fixed — no raw HTML issue)
# ─────────────────────────────────────────────
def render_result(predicted_idx, probs):
    predicted_class = CLASS_NAMES[predicted_idx]
    predicted_icon  = CLASS_ICONS[predicted_idx]
    confidence      = probs[predicted_idx] * 100

    # ── Result Header ──
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Predicted Class</div>
        <div class="result-class">{predicted_icon} {predicted_class}</div>
        <div class="result-confidence">Confidence: {confidence:.2f}%</div>
        <hr class="divider">
        <div class="result-label" style="margin-bottom:1rem;">All Probabilities</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability Bars — rendered one by one ──
    for i, (name, icon, prob) in enumerate(zip(CLASS_NAMES, CLASS_ICONS, probs)):
        pct        = prob * 100
        fill_class = "prob-fill-high" if i == predicted_idx else (
                     "prob-fill-medium" if pct > 5 else "prob-fill-low")
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-header">
                <span class="prob-name">{icon} {name}</span>
                <span class="prob-value">{pct:.2f}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill {fill_class}" style="width:{pct:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  UI — HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI Document Classifier</div>
    <h1>Identify Documents<br><span>Instantly & Accurately</span></h1>
    <p>Upload any document image and our deep learning model will classify it in milliseconds with 99% accuracy.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-number">99%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">3</div>
        <div class="stat-label">Document Types</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">150</div>
        <div class="stat-label">Test Images</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">CNN</div>
        <div class="stat-label">Model Type</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  UI — UPLOAD
# ─────────────────────────────────────────────
st.markdown("""
<div class="upload-section">
    <div class="upload-icon">📂</div>
    <div class="upload-title">Drop your document image here</div>
    <div class="upload-sub">Supports JPG, JPEG, PNG · Max 10MB</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="upload",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ─────────────────────────────────────────────
#  UI — PREVIEW + PREDICT
# ─────────────────────────────────────────────
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Document", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍  Classify Document"):
        with st.spinner("Analysing document..."):
            try:
                model, device  = load_model()
                pred_idx, probs = predict(image, model, device)
                render_result(pred_idx, probs)

            except FileNotFoundError:
                st.error("⚠️ Model file not found at output/model.pt")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

else:
    st.markdown("""
    <div style="
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        color: #94a3b8;
        font-size: 0.88rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    ">
        📌 Upload an image above to get started
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  UI — CLASS INFO CARDS
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex; gap:0.8rem; margin-top:0.5rem;">
    <div style="flex:1; background:#ffffff; border:1px solid #e2e8f0;
                border-radius:14px; padding:1.2rem; text-align:center;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size:1.8rem; margin-bottom:0.4rem;">🪪</div>
        <div style="font-family:'Syne',sans-serif; font-size:0.82rem; font-weight:600;
                    color:#1e3a5f; margin-bottom:0.2rem;">Driving License</div>
        <div style="font-size:0.72rem; color:#94a3b8;">Government issued driving permit</div>
    </div>
    <div style="flex:1; background:#ffffff; border:1px solid #e2e8f0;
                border-radius:14px; padding:1.2rem; text-align:center;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size:1.8rem; margin-bottom:0.4rem;">📄</div>
        <div style="font-family:'Syne',sans-serif; font-size:0.82rem; font-weight:600;
                    color:#1e3a5f; margin-bottom:0.2rem;">Others</div>
        <div style="font-size:0.72rem; color:#94a3b8;">Any other document type</div>
    </div>
    <div style="flex:1; background:#ffffff; border:1px solid #e2e8f0;
                border-radius:14px; padding:1.2rem; text-align:center;
                box-shadow:0 1px 4px rgba(0,0,0,0.06);">
        <div style="font-size:1.8rem; margin-bottom:0.4rem;">🔐</div>
        <div style="font-family:'Syne',sans-serif; font-size:0.82rem; font-weight:600;
                    color:#1e3a5f; margin-bottom:0.2rem;">Social Security</div>
        <div style="font-size:0.72rem; color:#94a3b8;">Social security card / document</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Footer ──
st.markdown("""
<div class="footer">
    Built with <span>PyTorch + Streamlit</span> · Document Classifier · 99% Accuracy
</div>
""", unsafe_allow_html=True)