"""Optional Dr7 text-only report draft client; never sends the eye image."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

API_URL = "https://dr7.ai/api/v1/medical/chat/completions"
DEFAULT_MODEL = "medgemma-4b-it"


def draft_report(result: dict[str, Any], examination: dict[str, Any], api_key: str, model: str = DEFAULT_MODEL) -> str:
    """Send only score/metadata JSON to Dr7 and return an unverified text draft.

    No patient name, date of birth, uploaded image, or original image filename is added.
    User-provided examination fields are intentionally limited to non-identifying labels.
    """
    if not api_key or not api_key.strip():
        raise ValueError("A Dr7 API key is required to request a draft.")
    payload_data = {
        "examination_information": {
            key: examination.get(key)
            for key in ("modality", "laterality")
            if examination.get(key)
        },
        "complete_cad_output": result,
        "image_transmitted": False,
    }
    system = (
        "You are drafting a report for a research eye-image classification prototype. "
        "Use only the supplied JSON. Do not infer image findings, symptoms, history, anatomy, "
        "or test results beyond the classifier fields. Do not convert model scores into clinical "
        "probabilities or a confirmed diagnosis. Preserve every class score and the score/calibration "
        "semantics. If information is absent, write 'Not provided'. Use exactly these headings: "
        "1. Examination information; 2. CAD findings; 3. Predicted disease and confidence score; "
        "4. Interpretation; 5. Recommendations for further evaluation. Keep recommendations to "
        "clinician review and state that this output is not diagnostic. The response is an unverified "
        "draft requiring clinician review; do not claim clinical validation."
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload_data, ensure_ascii=False, separators=(",", ":"))},
        ],
        "max_tokens": 900,
        "temperature": 0,
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key.strip()}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Dr7 API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Dr7 API: {exc.reason}") from exc
    try:
        text = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Dr7 response did not match the documented chat-completion structure.") from exc
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Dr7 returned an empty draft.")
    return text.strip()
