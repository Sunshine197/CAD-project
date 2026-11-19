from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import EyeDiseasePredictor
from reporting import build_report

st.set_page_config(page_title="Eye CAD Research Prototype", page_icon="", layout="wide")
st.title("Eye CAD · Research Prototype")
st.caption("Three-class image classification with transparent scores and a source-bound report.")
st.warning(
    "Research use only. This prototype is not a medical device or diagnostic system. "
    "Model scores are not confirmed-diagnosis probabilities. A qualified clinician must review all outputs.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Model")
    fresh_default = Path("fresh_group_split_run/fresh_group_split_model.pth")
    default_checkpoint = str(fresh_default) if fresh_default.is_file() else "best_eye_disease_model.pth"
    checkpoint_path = st.text_input("Checkpoint path", value=default_checkpoint, help="The fresh group-split checkpoint remains local and is not included in the public repository.")
    st.caption("Fresh locally retrained weights are preferred when present; otherwise the app uses the supplied starter checkpoint.")
    st.divider()
    st.header("Examination information")
    exam_date = st.date_input("Examination date (optional)", value=None)
    modality = st.text_input("Image type / modality (optional)", placeholder="Not provided")
    laterality = st.selectbox("Laterality", ["Not provided", "Right", "Left", "Both", "Unknown"])
    device = st.text_input("Imaging device/source (optional)", placeholder="Not provided")

uploaded = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
if not uploaded:
    st.info("Upload an image to run the classifier. The image is processed in this app session and is not added to the repository.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except (UnidentifiedImageError, OSError) as exc:
    st.error(f"The uploaded file could not be read as an image: {exc}")
    st.stop()

left, right = st.columns([1, 1.25], gap="large")
with left:
    st.image(image, caption="Uploaded image", width="stretch")
    st.caption(f"Input dimensions: {image.width} × {image.height} px")

@st.cache_resource(show_spinner="Loading classifier…")
def load_predictor(path: str) -> EyeDiseasePredictor:
    return EyeDiseasePredictor(path)

try:
    predictor = load_predictor(str(Path(checkpoint_path).expanduser()))
    result = predictor.predict(image)
except Exception as exc:
    st.error(f"Could not load the checkpoint or classify the image: {exc}")
    st.stop()

examination = {
    "date": exam_date,
    "modality": modality,
    "laterality": laterality if laterality != "Not provided" else "",
    "device": device,
}
report = build_report(result, examination)
with right:
    st.subheader("Classifier output")
    st.write("Scores are shown for all supported classes. They are not clinical probabilities.")
    for entry in result["predictions"]:
        st.progress(float(entry["score"]), text=f"{entry['class'].title()} — {100 * entry['score']:.1f}% model score")
    with st.expander("Full inference output and provenance", expanded=False):
        st.json(result)
    with st.expander("Unadjusted softmax scores"):
        st.json(result["raw_softmax_scores"])

st.subheader("Structured report")
st.code(report, language=None)
st.download_button("Download deterministic report (.txt)", report, file_name="eye_cad_report.txt", mime="text/plain")

st.divider()
st.subheader("Optional Dr7 AI draft")
st.write("The deterministic report above is the default. Dr7 can optionally produce a separate language-model draft using the complete score and metadata object.")
st.warning(
    "When enabled, the app sends the classifier scores, calibration status, model/preprocessing/checkpoint metadata, and non-identifying examination fields to Dr7. It does not send the image or patient name/date of birth. Dr7's published disclaimer describes the models as research-purpose and not clinical-grade. Review Dr7's current privacy and service terms before sending any sensitive or identifiable information.",
    icon="⚠️",
)
with st.form("dr7_form"):
    enabled = st.checkbox("I understand and want to send this text-only payload to Dr7 for an unverified draft.")
    api_key = st.text_input("Dr7 API key", type="password", help="Used only for this request; not stored in this app or repository.")
    model_name = st.selectbox("Dr7 model", ["medgemma-4b-it", "medgemma-27b-it"], index=0)
    submitted = st.form_submit_button("Generate Dr7 draft")
if submitted:
    if not enabled:
        st.warning("Check the opt-in box before sending a request to Dr7.")
    elif not api_key.strip():
        st.error("Enter a Dr7 API key to continue.")
    else:
        try:
            from dr7_client import draft_report
            with st.spinner("Requesting an unverified draft from Dr7…"):
                llm_report = draft_report(result, examination, api_key, model_name)
            st.subheader("Dr7-generated draft — unverified")
            st.error("Check every statement against the source data. Language models can invent or omit information; this draft is not medically validated and requires clinician review.")
            st.markdown(llm_report)
            st.download_button("Download Dr7 draft (.txt)", llm_report, file_name="eye_cad_dr7_draft.txt", mime="text/plain")
        except Exception as exc:
            st.error(f"Dr7 request failed: {exc}")

st.caption("No patient name or date of birth is collected. For an academic demonstration only; do not upload identifiable patient data.")
