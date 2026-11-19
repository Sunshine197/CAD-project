# Eye CAD report workflow implementation plan

## Approach

Keep the existing ResNet50 predictor and its matched 224×224 RGB/ImageNet-normalized preprocessing. The Streamlit app loads a checkpoint selected by path, displays every class score and full provenance, and creates a deterministic report using only the classifier output and explicitly entered examination fields. Missing information is written as “Not provided.” Model scores are called scores, never confirmed-diagnosis probabilities.

The default report is generated locally without a language model. An optional Dr7 adapter follows the currently published medical chat completion API and sends the complete model result plus non-identifying exam fields as JSON text. It never sends the eye image or patient identifiers. The user must explicitly enable the request and enter their API key for that session. Any response is shown as an unverified draft, not automatically validated, and requires clinician review. No clinical-grade or hallucination-free claim is made.

## Structure

- `inference.py`: model loading, preprocessing, prediction, calibration and provenance.
- `reporting.py`: deterministic, source-bound report composition.
- `dr7_client.py`: optional text-only Dr7 request.
- `app.py`: Streamlit interface, upload, score display, report preview and export.
- `DR7_INTEGRATION_NOTES.md`: source-grounded API and privacy/validation limitations.

The local Streamlit application uses the existing Python runtime and checkpoint files. No dataset, filename split manifest, or fresh model weights are added to the public repository.

## Design

- **Exploratory Clinical Console** (0.07): compact evidence panel, sober colors, dense audit metadata.
- **Academic Lab Notebook** (0.05): research annotations, muted paper tones, modular cards.
- **Diagnostic Control Room** (0.08): high-contrast instrumentation and workflow status.

Chosen: **Exploratory Clinical Console**. Use an understated clinical palette and clear information hierarchy. Distinguish classifier output from clinician interpretation. Scores, calibration status, model provenance, and limitations remain visible and downloadable. Interaction is explicit: upload, inspect, and optionally request an external-language-model draft. Avoid celebratory success states or UI that implies diagnosis. Animation is minimal and limited to Streamlit's standard progress feedback. Typography uses Streamlit's readable system sans-serif with restrained heading hierarchy. Brand essence: a transparent academic eye-image CAD demonstrator for student/research review; cautious, inspectable, practical. Brand voice is direct and uncertainty-aware (e.g., “Model scores—not diagnostic probabilities”; “Review every draft against the source data”). Wordmark is text-only “Eye CAD / Research Prototype,” no clinical seal or logo. Signature color is deep teal for neutral emphasis, never as a disease indicator.
