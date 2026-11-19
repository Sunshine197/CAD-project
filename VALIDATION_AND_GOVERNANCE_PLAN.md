# External validation and governance plan

**Status:** Planning only. No independent clinical validation has been performed. This plan does not authorize patient-care use or establish regulatory compliance.

## Current evidence and release boundary

The current local checkpoint has an 85.45% accuracy result from a filename-stem-grouped internal split. Patient-level independence is unverified because the mixed-source archive has no source-to-patient manifest. Two exact image contents appear under different class labels. The test set is therefore exploratory, and the model's temperature-scaled outputs are not established clinical probabilities. See `DATASET_GROUPING_REVIEW.md` and the README.

The GUI upload path is smoke-tested, but interface behavior is not evidence of diagnostic performance. Qualified eye-care professional review is still pending. Dr7's published terms prohibit clinical decision use and processing real patient data; do not send real-patient-derived outputs to Dr7 unless Dr7 resolves the terms conflict in writing and authorizes the exact research use. See `DR7_INTEGRATION_NOTES.md`.

Until the gates below are met, keep this as a research prototype. Do not use its output to diagnose, rule out disease, triage, recommend treatment, or make patient-care decisions. “Normal” must not be presented as a comprehensive normal eye examination.

## Phased plan

### Gate 1 — Intended use and clinician review

Keep the intended use narrow and written down: research evaluation of a three-label retinal-image classifier, not diagnosis or clinical decision support. Obtain the qualified eye-care professional's completed `CLINICIAN_REVIEW_CHECKLIST.md`. Resolve the meaning and unit of each label (image, eye, or person), whether the three labels can overlap, and how to phrase the unsupported-class and “normal” limitations. Record every requested change and keep the reviewer's identity and any patient information out of the public repository.

**Exit condition:** clinician feedback is documented and all required prototype wording changes are made. This is a review gate, not validation or approval for clinical use.

### Gate 2 — Data provenance, rights, and governance

Before collecting another dataset, identify the data owner and source for every image, confirm permission for research and publication, and obtain an institutional ethics/IRB determination or other required governance approval. Use a data-use agreement where required. Maintain a restricted source manifest that maps each image to its source and patient/eye identifier; do not commit this manifest or any images to GitHub. Determine whether both eyes, repeated visits, and near-duplicate images can be linked. Resolve the existing cross-class exact duplicates with the source owner or qualified graders rather than choosing a label based on model behavior.

If the original source-to-patient mapping cannot be obtained, do not claim patient-independent evaluation from filename grouping. Find an independent cohort with verified patient-level grouping instead.

**Exit condition:** written data permissions and governance determination are in place, labels and analysis unit are defined, and the cohort can be separated by verified patient identity and source.

### Gate 3 — Freeze the study before external testing

Version and freeze the model checkpoint, class mapping, resize/color/normalization pipeline, score calculation, report template, and analysis plan before opening the external test results. Record code and model hashes in a controlled, non-sensitive study record. Do not use the external test data for model selection, threshold tuning, calibration, or prompt iteration. If calibration is needed, reserve a separate calibration cohort; keep the final external test untouched.

Predefine the primary endpoint and sample-size target based on the desired confidence-interval precision, with clinician and statistical input. State the unit of analysis and handling of indeterminate, unreadable, multi-label, and missing-reference cases before evaluation.

**Exit condition:** signed-off protocol and frozen model/analysis code, with a genuinely independent final-test cohort.

### Gate 4 — Reference standard and performance evaluation

Use qualified eye-care professionals to establish reference labels under a written grading protocol. Where feasible, obtain independent masked grading by at least two qualified graders and adjudicate disagreements. Graders should not see model predictions while creating reference labels. State exactly whether the target is an image, eye, or patient; do not convert image-level folders into diagnoses without appropriate reference evidence.

Report a confusion matrix and per-class sensitivity and specificity with confidence intervals, plus suitable class-balanced summary measures. Include one-vs-rest discrimination metrics where the label structure supports them. Assess calibration separately using a reliability plot and proper scoring rules such as Brier score and log loss; do not call scores probabilities unless calibration is adequate for the intended population and use. Report performance by source/site and relevant subgroups only where data are appropriately collected and sample sizes support interpretable estimates. Include image-quality failures, abstentions, missing labels, and exclusions in the flow accounting.

**Exit condition:** an independent analyst reproduces the prespecified analysis and clinicians review whether the estimates and uncertainty are meaningful for the defined research question.

### Gate 5 — Report and optional language-model evaluation

Evaluate the deterministic report separately from classifier accuracy. Use a clinician-authored rubric to check that each statement is grounded in supplied fields, missing information stays marked as missing, model scores are not described as diagnoses, uncertainty is clear, and recommendations do not exceed the evidence. Record unsupported statements, omitted limitations, and potentially misleading language.

Do not test Dr7 with real patient data under its current published terms. If a permitted synthetic/non-patient test is conducted, treat it only as API/interface testing, not a medical-quality evaluation. A later real-data study requires prior written vendor clarification and an agreement that expressly covers the exact research purpose, data handling, and use restrictions. Keep LLM drafts separate from the classifier's ground truth and require human review; do not rely on prompts to guarantee factuality.

**Exit condition:** report-generation review has a prespecified rubric, qualified reviewers, and documented failure criteria. Any external LLM use has separate data and vendor approval.

### Gate 6 — Deployment decision and monitoring

Do not make a clinical-use or deployment decision from the current dataset or internal scores. Any future proposed clinical use requires separate review of intended purpose, local legal/regulatory obligations, security, privacy, workflow risks, user training, human oversight, and post-deployment monitoring by qualified institutional stakeholders. Establish incident reporting, version control, drift review, and a rollback plan before any authorized deployment. Obtain professional regulatory/legal advice where needed; this project plan is not a regulatory determination.

**Exit condition:** all required clinical, data-governance, security, and regulatory approvals are documented for the exact intended use. Until then, no clinical deployment.

## Current action order

1. Keep the clinician review request open and apply any feedback when received.
2. Ask for source-to-patient mapping and permission records; if unavailable, source a separate governed cohort rather than relabeling the current score as patient-independent.
3. Resolve the two cross-class exact duplicates with a source owner or qualified graders.
4. Freeze a study protocol and external test plan before obtaining or examining external test predictions.
5. Keep Dr7 disconnected from real-patient-derived inputs while its published terms remain unresolved.

These steps are prerequisites for a future evaluation, not evidence that the prototype is safe or clinically ready.
