# GUI smoke test

**Status:** Completed 19 November 2026. This is an interface smoke test only, not clinical validation, a performance evaluation, or a live Dr7 integration test.

The Streamlit upload flow was exercised with one local example image from each retained class folder in the supplied dataset. The images and filenames were not added to GitHub, and no images or examination information were sent to Dr7. The fresh group-split checkpoint available in the local workspace was selected automatically.

For all three uploads, the app ran without an application exception and rendered three class-score bars, the full output/provenance JSON (including raw scores and calibration metadata), the structured report with all five requested sections, and the deterministic report download control. The report displayed the research-only warning, the unverified patient-grouping caveat, and the duplicate-label concern. The optional Dr7 form did not generate a draft because it was not submitted.

These checks show that the tested local UI path functions. They do not show that any prediction is correct, that the score is a clinical probability, that the input examples are representative, that patient-level independence is established, or that the Dr7 request works with a real account. The downloadable control was present in the rendered interface; no external report service was used.
