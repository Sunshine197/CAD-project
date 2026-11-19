# Dataset grouping audit

**Status as of 19 November 2026: source confirmed by the user; patient-level grouping is not confirmed.** The user confirmed that the supplied archive came from the Kaggle listing linked below.

The confirmed Kaggle listing describes a four-class retinal-image collection assembled from multiple sources, naming IDRiD, “Ocular recognition,” and HRF among them. It does not provide a per-image source key or a subject-ID manifest. [1]

The original ODIR-2019 documentation says its records cover 5,000 patients with color fundus photographs of both eyes. It states that ODIR labels are based on both eye images and patient age, and that a patient may have one or more labels. A separate Scientific Reports study likewise describes 10,000 ODIR photographs from 5,000 patients. This confirms patient-level pairing for ODIR itself, but it does **not** prove that every numeric `_left`/`_right` filename in this mixed-source Kaggle derivative is an ODIR patient ID. [2] [3]

The supplied archive contains no README, CSV, or source-to-patient mapping. In the 2,918 JPGs from the three retained classes, there are 2,249 distinct stems after removing `_left`/`_right`; 667 stems have both suffixes. Three stems occur in more than one class folder. Two exact image copies occur in different class folders (cataract and glaucoma). Both copies were assigned to the same partition as their duplicate during the completed run, and a SHA-256 comparison found no exact duplicate image across partitions. The conflicting class folders still create a label-quality concern; this audit does not decide which label is correct.

Therefore, the completed 85.45% result is from a **filename-stem-grouped internal split**, not a verified patient-independent test. The grouping avoided splitting matching stems and the detected exact duplicates across partitions, but without source metadata it cannot rule out patient overlap under different IDs or source-specific identifier collisions. Do not describe it as confirmed patient-level validation.

## Next evidence needed

Confirm that the archive came from [the matching Kaggle listing](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) and, if available, obtain the original source manifest that maps each image to its source and patient/record ID. If that mapping is not available, preserve the current filename-based description and treat the internal score as exploratory. Review how to handle the two identical images appearing in distinct class folders before relying on class-specific performance.

## References

[1]: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification "Eye Diseases Classification dataset page"
[2]: https://odir2019.grand-challenge.org/dataset/ "ODIR-2019 dataset documentation"
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10352300/ "Population-independent multi-disease detection study"
