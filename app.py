import io
import json
import os
import cv2
import gdown
import numpy as np  # Ensure numpy is explicitly imported here
from PIL import Image
import google.generativeai as genai
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# -------------------------------------------------------------
# Page Configuration & Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Retinal Pathology Hybrid Diagnostic System",
    page_icon="👁️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .recommendation-box {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 1.8rem;
        margin-top: 1rem;
        border-left: 6px solid #2563EB;
        border: 1px solid #E2E8F0;
        color: #1E293B;
    }
    .recommendation-box h3 {
        color: #1E3A8A;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .logo-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1rem;
    }
    .logo-badge {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        padding: 8px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    </style>
""",
    unsafe_allow_html=True,
)

CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

CLINICAL_DATABASE = {
    "cataract": {
        "description": "Clouding of the crystalline lens. Manifests as global opacity and obscured retinal margins.",
        "findings": "Visual inspection reveals global opacification across the crystalline lens, resulting in diminished fundus reflex clarity and obscured posterior pole margins.",
        "action_plan": (
            "• **Confirmatory Tests Required:** Slit-lamp biomicroscopy, visual acuity assessment, and keratometry for intraocular lens calculation.<br><br>"
            "• **Triage Urgency Level:** Routine / Semi-Urgent.<br><br>"
            "• **Patient Recommendations:** Schedule an ophthalmology consultation for surgical evaluation (phacoemulsification)."
        ),
    },
    "diabetic_retinopathy": {
        "description": "Microvascular retinal complications. Hallmark signs: microaneurysms, hemorrhages, and hard exudates.",
        "findings": "Fundus morphology demonstrates hallmark microvascular complications, including scattered microaneurysms, intraretinal dot-and-blot hemorrhages, and lipid hard exudates.",
        "action_plan": (
            "• **Confirmatory Tests Required:** Optical Coherence Tomography (OCT) to evaluate for macular edema; Fluorescein Angiography (FA).<br><br>"
            "• **Triage Urgency Level:** Semi-Urgent to Urgent Specialist Referral.<br><br>"
            "• **Patient Recommendations:** Immediate consultation with a retinal specialist; strict glycemic control."
        ),
    },
    "glaucoma": {
        "description": "Progressive optic neuropathy characterized by optic disc cupping and neuroretinal rim thinning.",
        "findings": "Evaluation indicates optic nerve head abnormalities, characterized by an enlarged cup-to-disc ratio and localized neuroretinal rim thinning.",
        "action_plan": (
            "• **Confirmatory Tests Required:** Goldmann applanation tonometry, Standard Automated Perimetry, and RNFL thickness mapping via OCT.<br><br>"
            "• **Triage Urgency Level:** Semi-Urgent.<br><br>"
            "• **Patient Recommendations:** Urgent glaucoma workup with an optometrist or ophthalmologist."
        ),
    },
    "normal": {
        "description": "Healthy retinal fundus with clear vascular architecture and physiological optic cup-to-disc ratio.",
        "findings": "Retinal fundus scan exhibits healthy morphology with clear, uninterrupted vascular arcades and well-defined optic disc margins.",
        "action_plan": (
            "• **Confirmatory Tests Required:** Standard routine comprehensive eye examination.<br><br>"
            "• **Triage Urgency Level:** Routine Preventive Care.<br><br>"
            "• **Patient Recommendations:** Maintain an annual screening schedule and balanced nutrition."
        ),
    },
}

# Configure Gemini API securely via Streamlit Secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


@st.cache_resource
def load_resnet_model(weights_path="best_resnet18_weights.pth"):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Automatically download from Google Drive using gdown if missing
  if not os.path.exists(weights_path):
    file_id = "1N6aJjKxRM7tBVSiMCJjy1Wj1zLed1csj"
    url = f"https://drive.google.com/uc?id={file_id}"
    with st.spinner(
        "📥 Downloading trained ResNet18 weights from secure Google Drive..."
    ):
      gdown.download(url, weights_path, quiet=False)

  model = models.resnet18(weights=None)
  in_features = model.fc.in_features
  model.fc = nn.Linear(in_features, len(CLASS_NAMES))

  if os.path.exists(weights_path):
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
  else:
    st.error(f"⚠️ Weights file '{weights_path}' missing!")

  model = model.to(device)
  model.eval()
  return model, device


model, device = load_resnet_model()


def crop_retina_circle(pil_img):
  np_img = np.array(pil_img)
  gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
  _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(
      thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  if contours:
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w > 50 and h > 50:
      return Image.fromarray(np_img[y : y + h, x : x + w])
  return pil_img


PREPROCESS = transforms.Compose([
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=True,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])

# Sidebar with Logo/Branding
with st.sidebar:
  st.markdown(
      """
        <div class="logo-container">
            <div class="logo-badge">👁️ RETINO-AI</div>
        </div>
    """,
      unsafe_allow_html=True,
  )
  st.title("Settings")
  st.markdown("""
    **System Overview:**
    * **Architecture:** Hybrid ResNet18 + Gemini Vision API
    * **Validation Accuracy:** 94.91%
    * **Screenings:** Cataract, Diabetic Retinopathy, Glaucoma, Normal
    """)
  st.success("⚡ Hybrid Diagnostic Mode Active")

# Main Page Header with Logo Badge
st.markdown(
    """
    <div class="logo-container">
        <div class="logo-badge">👁️ RetinoAI Studio</div>
        <span style="color: #64748B; font-weight: 600; font-size: 0.95rem;">Advanced Clinical Decision Support</span>
    </div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">Automated Retinal Diagnostic System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Hybrid clinical decision support via ResNet'
    ' feature extraction and Gemini clinical verification.</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload patient retinal fundus scan (JPG, PNG)...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
  raw_image = Image.open(uploaded_file).convert("RGB")
  cleaned_image = crop_retina_circle(raw_image)

  # 1. Base ResNet Inference
  tensor = PREPROCESS(cleaned_image).unsqueeze(0).to(device)
  with torch.no_grad():
    outputs = model(tensor)
    probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]

  top_idx = int(probabilities.argmax())
  resnet_class = CLASS_NAMES[top_idx]
  resnet_conf = float(probabilities[top_idx]) * 100.0
  resnet_distribution = {
      CLASS_NAMES[i]: round(float(probabilities[i]) * 100.0, 2)
      for i in range(len(CLASS_NAMES))
  }

  col1, col2 = st.columns([1, 1.3], gap="large")

  with col1:
    st.subheader("Patient Fundus Scan")
    st.image(
        cleaned_image,
        use_container_width=True,
        caption=f"File: {uploaded_file.name}",
    )

  with col2:
    st.subheader("Hybrid Diagnostic Assessment")

    # 2. Gemini Multimodal Cross-Verification Call
    with st.spinner(
        "🤖 Running Gemini + ResNet Hybrid Clinical Cross-Verification..."
    ):
      prompt = f"""
            You are an expert ophthalmology AI assistant working alongside a deep learning model.
            The local ResNet18 model analyzed this retinal scan and predicted: '{resnet_class}' with a confidence of {resnet_conf:.2f}%.
            Full ResNet distribution: {resnet_distribution}

            Please examine this retinal image directly, cross-reference it with the model's prediction, and provide a final hybrid assessment.
            Respond strictly in valid JSON format with keys:
            "final_prediction" (one of: cataract, diabetic_retinopathy, glaucoma, normal),
            "hybrid_confidence_score" (float between 0 and 100),
            "clinical_reasoning" (string explanation).
            """
      try:
        # Fixed model name to gemini-1.5-pro for stable API cross-verification
        gemini_model = genai.GenerativeModel("gemini-1.5-pro")
        response = gemini_model.generate_content([cleaned_image, prompt])

        # Clean and parse JSON response text from Gemini
        cleaned_text = (
            response.text.replace("```json", "").replace("```", "").strip()
        )
        hybrid_data = json.loads(cleaned_text)

        final_pred = (
            hybrid_data.get("final_prediction", resnet_class)
            .lower()
            .strip()
        )
        hybrid_conf = float(
            hybrid_data.get("hybrid_confidence_score", resnet_conf)
        )
        clinical_notes = hybrid_data.get(
            "clinical_reasoning", "Validated via hybrid consensus."
        )
      except Exception as e:
        # Fallback safely to ResNet standalone result if API quota or parsing hits an issue
        final_pred = resnet_class
        hybrid_conf = resnet_conf
        clinical_notes = (
            "ResNet primary inference used directly (API fallback active:"
            f" {str(e)})"
        )

    badge_color = "#16A34A" if final_pred == "normal" else "#DC2626"
    st.markdown(
        f"""
        <div style="background-color:#F8FAFC; border-left:6px solid {badge_color}; padding:15px; border-radius:8px; border: 1px solid #E2E8F0;">
            <h4 style="margin:0; color:#374151;">Hybrid Verified Classification:</h4>
            <h2 style="margin:4px 0; color:{badge_color}; text-transform:capitalize;">{final_pred.replace('_', ' ')}</h2>
            <p style="margin:0; color:#4B5563; font-weight:600;">Consensus Confidence: {hybrid_conf:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.info(f"**Gemini Clinical Synthesis:** {clinical_notes}")

  # Clinical Recommendations & Action Plan Section
  st.divider()
  st.subheader("Clinical Recommendations & Action Plan")

  # Match key safely
  matched_key = final_pred if final_pred in CLINICAL_DATABASE else "normal"
  report = CLINICAL_DATABASE[matched_key]

  st.markdown(
      f"""
    <div class="recommendation-box">
        <h3>1. Findings Analysis</h3>
        <p>{report['findings']}</p>
        <hr style="margin: 1.2rem 0; border: 0; border-top: 1px solid #E2E8F0;">
        <h3>2. Recommended Action Plan</h3>
        <div style="line-height: 1.6;">{report['action_plan']}</div>
    </div>
    """,
      unsafe_allow_html=True,
  )
