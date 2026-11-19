# Clinician review checklist — Eye CAD research prototype

**Review status:** Pending review by a qualified eye-care professional. This checklist is not a clinical validation study and does not authorize patient-care use.

## Prototype under review

This research prototype classifies an uploaded retinal/fundus image into one of three training labels: cataract, glaucoma, or normal. It displays raw softmax outputs and, for the locally retrained checkpoint, temperature-scaled model scores using a calibration parameter fitted on a reserved partition of the supplied dataset. The screen and report explicitly say that scores are not clinically validated probabilities, the prediction is not a confirmed diagnosis, and clinician review is required. The report template does not infer symptoms, history, or examination findings that were not entered.

The user's confirmed source is Kaggle's `gunavenkatdoddi/eye-diseases-classification`. Its description says the collection includes images from multiple sources. The archive supplied here did not contain a per-image source or patient-ID map. Two exact image contents occur in more than one class folder. The current split groups filename stems, but patient-level independence is not verified. See `DATASET_GROUPING_REVIEW.md`.

## Questions for the reviewer

For each item, mark **Accept**, **Revise**, or **Unable to assess**, then add a short rationale.

1. **Intended-use boundary:** Is the current “research prototype only; not for diagnosis or treatment” boundary prominent and clear enough for an academic demonstration? What stronger limitation should appear?

   Decision: ___  Notes: ______________________________________________

2. **Label meaning:** Do the folders `cataract`, `glaucoma`, and `normal` represent eye-level diagnoses, patient-level labels, or image-level categories in this mixed-source dataset? Is “normal” defined sufficiently to justify that label? Is the source's ODIR component's patient-level/multi-label convention correctly distinguished from the derived single-class folders?

   Decision: ___  Notes: ______________________________________________

3. **Supported conditions and omissions:** Is it clear that the classifier only recognizes these three labels and cannot rule out other ocular disease or image-quality problems? Should “normal” be presented differently to avoid implying a comprehensive normal exam?

   Decision: ___  Notes: ______________________________________________

4. **Score language:** Are “model score” and “not a clinical probability” sufficiently clear? Is any use of the word “confidence” misleading without clinically validated calibration or decision thresholds?

   Decision: ___  Notes: ______________________________________________

5. **Report wording:** Are the five sections—examination information, CAD findings, predicted class/scores, interpretation, and further evaluation—appropriately cautious? Does any sentence overstate what the image classifier establishes?

   Decision: ___  Notes: ______________________________________________

6. **Recommendations:** Is the generic recommendation to seek qualified clinician review appropriate? What wording should be removed or added? Please do not use this checklist to create patient-specific medical advice.

   Decision: ___  Notes: ______________________________________________

7. **Dataset quality:** How should two exact image contents assigned to different class folders be handled? What source labels or expert adjudication would be required before using class-specific performance results?

   Decision: ___  Notes: ______________________________________________

8. **Validation and deployment:** What additional independent, patient-linked data, reference standard, subgroup analysis, and governance would be needed before making any clinical-use claim? Does the current filename grouping support any patient-independent performance claim? (Current documented answer: no.)

   Decision: ___  Notes: ______________________________________________

9. **External LLM draft:** The optional Dr7 output is a text-only, unverified draft. Does the app's warning sufficiently prevent users from treating it as a diagnosis or validated report? What review controls should remain mandatory?

   Decision: ___  Notes: ______________________________________________

## Reviewer conclusion

Overall: ☐ Suitable only for an academic demonstration with stated limits  ☐ Revise before demonstration  ☐ Do not use in current form

Required changes before any demonstration: __________________________________________

This review is not proof of clinical validity, regulatory compliance, or safety for patient care. Do not include patient names, dates of birth, images, or other identifiable information in this form or in GitHub.

## References for source/context

- Kaggle source page: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification
- ODIR-2019 dataset documentation: https://odir2019.grand-challenge.org/dataset/
- ODIR population/dataset study: https://pmc.ncbi.nlm.nih.gov/articles/PMC10352300/
