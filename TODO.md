# TODO

- [x] Deterministic structured report: Include examination information; CAD findings; predicted disease and confidence score; interpretation; recommendations for further evaluation. Use only supplied data. Missing examination fields say “Not provided.” Do not invent findings, present a prediction as confirmed diagnosis, or call model scores clinical probabilities. Communicate the internal-only calibration limitation and require qualified clinician review.
- [x] Interactive GUI: Upload an image; load a selected checkpoint; display every supported class score, raw scores, metadata, provenance, calibration status, limitations, deterministic report preview, and text download. Keep patient identifiers out of the form.
- [x] Optional Dr7 pathway: Send the complete classifier output and non-identifying examination metadata, not only the top predicted disease. Require explicit opt-in and user-entered API key; do not send the image. Label returned text as an unverified draft and warn that no prompt guarantees a hallucination-free or clinically validated report.
- [ ] Obtain and assess applicable Dr7 privacy, service, and data-processing terms before any use with sensitive/identifiable information.
- [ ] Arrange independent clinical review and appropriately governed external validation before any clinical-use claims or deployment.
