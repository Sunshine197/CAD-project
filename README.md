# Eye disease CAD inference prototype

This folder contains a reusable inference module reconstructed from `image-based-eye-diseases-detection-final.ipynb` and the supplied `best_eye_disease_model.pth` checkpoint.

## Install and launch the interactive app

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Choose a checkpoint in the sidebar (the fresh group-split checkpoint is available locally as `fresh_group_split_run/fresh_group_split_model.pth`, but is not included in the public repository). Upload an image to see every class score and model provenance, preview a deterministic report, and download it as text. The app does not collect patient names or dates of birth.

The local deterministic report is the default and is composed from the classifier output plus examination fields explicitly entered by the user. Missing fields are marked “Not provided.” Recommendations are limited to clinician review. The report does not invent symptoms or exam findings and does not treat the classifier prediction as a confirmed diagnosis.

### Optional Dr7 draft

The app includes an explicitly opt-in Dr7 medical chat completion call. It sends the full classifier output, including all class scores and model/calibration/preprocessing/checkpoint metadata, plus modality and laterality when entered. It does **not** send the image, patient name, date of birth, examination date, device field, or original filename. The API key is supplied for the request and is not stored by the app. See [Dr7 integration notes](DR7_INTEGRATION_NOTES.md) and review Dr7's current terms and privacy practices before use. Dr7's published disclaimer describes its models as research-purpose and not clinical-grade. An LLM response is shown as an unverified draft requiring clinician review; no prompt or LLM can guarantee a hallucination-free report.

## Run a prediction

```python
from inference import EyeDiseasePredictor

predictor = EyeDiseasePredictor("best_eye_disease_model.pth")
result = predictor.predict_file("fundus_image.jpg")
print(result)
```

The result includes all class softmax scores, model/checkpoint metadata, and limitations. The score order is `cataract`, `glaucoma`, `normal`, matching the notebook's sorted class mapping. Evaluation preprocessing is resize to 224×224, RGB conversion, tensor conversion, and ImageNet mean/std normalization.

## Step 3: held-out evaluation and calibration

Run from this directory in an environment where the dataset is mounted:

```bash
python evaluate.py \
  --dataset /kaggle/input/eye-diseases-classification/dataset \
  --checkpoint best_eye_disease_model.pth \
  --output evaluation_metrics.json
```

The script creates a deterministic stratified 60/20/20 image-level re-split (`random_state=42`), fits one temperature-scaling parameter on that re-split's validation data, and reports aggregate metrics. It excludes per-image scores and image paths from the JSON output. **It cannot reconstruct the notebook's exact splits unless the original path ordering or split manifest is available.** Since the checkpoint was trained/selected from this dataset, a re-split may overlap training or checkpoint-selection examples; those metrics are exploratory, not independent test/calibration evidence.

The uploaded archive contained 938 cataract JPGs, 906 glaucoma JPGs, and 1,074 normal JPGs, matching the notebook's JPG counts; additional PNG files are ignored to match training preprocessing. The sorted-path re-split produced 84.93% test accuracy, compared with 88.53% recorded in the notebook. Because these differ, the original test assignments have not been reproduced. Do not cite the re-split result as the model's held-out test performance. For valid calibration/evaluation, use the original validation/test sample assignments or a genuinely independent, appropriately governed dataset.

## Fresh group-aware training when original split IDs are missing

The existing checkpoint cannot be fairly evaluated on a newly created split of the same source images. `train_fresh.py` provides an internal alternative: it starts from ImageNet ResNet50 weights (not the existing CAD checkpoint), reserves about 60%/10%/10%/20% for training/model selection/calibration/final test, and keeps matching filename stems together after removing `_left`/`_right`. This is a **filename-grouped split, not a confirmed patient-level split**. The matching Kaggle dataset page describes images from multiple sources and the supplied archive has no source/patient manifest, so the filename-to-patient assumption remains unverified. It writes the new checkpoint and aggregate metrics to a local output directory; split records and image data are not added to GitHub.

```bash
python train_fresh.py \
  --dataset /path/to/dataset \
  --output-dir fresh_group_split_run \
  --epochs 25 \
  --device cpu
```

This produces a new internally evaluated prototype, not external or clinical validation. For this run, the filename-stem-grouped final-test accuracy was 85.45%; class recall was 95.21% for cataract, 80.77% for glaucoma, and 80.84% for normal. Temperature scaling fitted on the separate internal calibration partition used T=1.55181; on the final-test partition it changed NLL from 0.36455 to 0.34266 and ECE from 0.05156 to 0.03797, while accuracy remained 85.45%. These are internal results on the supplied dataset only. Two exact image copies were found under different class labels; both copies were kept in the same split, but the conflicting labels remain a data-quality concern. The public source describes a mixed-source dataset and no per-image patient mapping was supplied, so this cannot be called patient-independent evaluation. See [the grouping audit](DATASET_GROUPING_REVIEW.md). Full aggregate metrics are in the local, ignored `fresh_group_split_run/fresh_group_split_metrics.json`; neither the fresh weights nor dataset/split records are published.

## Validation performed

The supplied checkpoint loaded successfully with strict state-dictionary matching using the reconstructed architecture. A smoke test on a synthetic blank image confirmed that inference returns all three classes and scores summing to approximately 1.0. That synthetic-image output is **not** a medical prediction and is not used as performance evidence.

## Important limitations

The fresh model's temperature was fitted on an internal calibration partition and its grouped final test result is documented in `fresh_group_split_run/fresh_group_split_metrics.json` locally. Temperature scaling improved aggregate internal test NLL and ECE but does not establish clinical calibration or external validity. Scores are model scores, not clinically validated disease probabilities. The original notebook reports 88.53% test accuracy on its own split; its original sample assignments were unavailable, so that result was not independently reproduced. This research prototype is not a medical device or confirmed diagnosis and must not substitute for a qualified clinician's judgment. Do not use it for clinical decisions.
