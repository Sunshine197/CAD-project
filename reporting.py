"""Deterministic, source-bound report text for the eye CAD research prototype."""
from __future__ import annotations

from datetime import date
from typing import Any


def _provided(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "Not provided."


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def build_report(result: dict[str, Any], examination: dict[str, Any] | None = None) -> str:
    """Build a report using only explicitly supplied fields and model output."""
    examination = examination or {}
    ranked = result.get("predictions", [])
    if not ranked:
        raise ValueError("The classifier result contains no predictions.")
    top = ranked[0]
    classes = result.get("metadata", {}).get("classes_in_logit_order", [])
    scores = {str(item["class"]): float(item["score"]) for item in ranked}
    display_names = {"cataract": "Cataract", "glaucoma": "Glaucoma", "normal": "Normal"}

    exam_date = examination.get("date") or "Not provided."
    if isinstance(exam_date, date):
        exam_date = exam_date.isoformat()

    class_rows = [f"- {display_names.get(name, name)}: {_pct(scores[name])}" for name in classes if name in scores]
    if not class_rows:
        class_rows = [f"- {display_names.get(str(item['class']), item['class'])}: {_pct(item['score'])}" for item in ranked]
    calibration = result.get("calibration", {})
    calibration_text = calibration.get("status", "Not specified.")
    score_semantics = result.get("score_semantics", "Model scores; not diagnostic probabilities.")

    interpretation = (
        f"The highest-ranked class is {display_names.get(str(top['class']), top['class'])}. "
        "This is the classifier's output for this image only. It does not establish or exclude disease, "
        "and the score should not be interpreted as a confirmed-diagnosis probability."
    )
    return "\n".join([
        "EYE IMAGE CAD REPORT — RESEARCH PROTOTYPE",
        "",
        "1. Examination information",
        f"- Examination date: {exam_date}",
        f"- Image type/modality: {_provided(examination.get('modality'))}",
        f"- Laterality: {_provided(examination.get('laterality'))}",
        f"- Imaging device/source: {_provided(examination.get('device'))}",
        "- Patient identifiers: Not collected by this prototype.",
        "",
        "2. CAD findings",
        f"- Automated image classifier: {result.get('metadata', {}).get('model', 'Not specified.')}",
        f"- Input image dimensions: {result.get('metadata', {}).get('input_image_size_px', 'Not specified.')}",
        f"- Preprocessing: RGB; resized to 224×224; ImageNet mean/std normalization.",
        "- The system returns scores for the three supported classes only: cataract, glaucoma, and normal.",
        "",
        "3. Predicted disease and confidence score",
        f"- Highest-ranked class: {display_names.get(str(top['class']), top['class'])} ({_pct(top['score'])} model score).",
        *class_rows,
        f"- Score description: {score_semantics}",
        f"- Calibration status: {calibration_text}",
        "",
        "4. Interpretation",
        interpretation,
        "The provided information does not include clinical history, symptoms, visual acuity, examination findings, or confirmatory testing. No such findings are inferred here.",
        "No clinically validated low/high score cutoff is established. A low score or close ranking must not be interpreted as a reliable rule-out or rule-in result.",
        "",
        "5. Recommendations for further evaluation",
        "- Have the image and classifier output reviewed by a qualified eye-care professional in the context of a comprehensive clinical assessment.",
        "- Do not use this output alone to make diagnostic or treatment decisions.",
        "",
        "Limitations: Research prototype only; not a medical device, confirmed diagnosis, or substitute for professional judgment. The dataset split groups filename stems, not verified patient IDs; patient-level independence is unknown. Exact duplicate images in different class folders create a label-quality concern. The model covers only the listed classes. Internal dataset calibration does not establish clinical calibration or performance on external data.",
        f"Model checkpoint SHA-256: {result.get('metadata', {}).get('checkpoint_sha256', 'Not available.')}",
    ])
