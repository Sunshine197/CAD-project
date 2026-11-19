# Eye disease CAD inference prototype

This folder contains a reusable inference module reconstructed from `image-based-eye-diseases-detection-final.ipynb` and the supplied `best_eye_disease_model.pth` checkpoint.

## Install

```bash
python -m pip install -r requirements.txt
```

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

The existing checkpoint cannot be fairly evaluated on a newly created split of the same source images. `train_fresh.py` provides a clean internal alternative: it starts from ImageNet ResNet50 weights (not the existing CAD checkpoint), reserves about 60%/10%/10%/20% for training/model selection/calibration/final test, and groups `_left`/`_right` images by filename ID so paired images do not cross partitions. It writes the new checkpoint and aggregate metrics to a local output directory; split records and image data are not added to GitHub.

```bash
python train_fresh.py \
  --dataset /path/to/dataset \
  --output-dir fresh_group_split_run \
  --epochs 25 \
  --device cpu
```

This produces a new internally evaluated prototype, not external or clinical validation. Filename-derived groups are a proxy for subjects and should be confirmed against the dataset documentation.

## Validation performed

The supplied checkpoint loaded successfully with strict state-dictionary matching using the reconstructed architecture. A smoke test on a synthetic blank image confirmed that inference returns all three classes and scores summing to approximately 1.0. That synthetic-image output is **not** a medical prediction and is not used as performance evidence.

## Important limitations

The softmax scores are not established as calibrated probabilities until evaluation has been run and reviewed. The original notebook reports 88.53% test accuracy on its own held-out test split; this has not been independently reproduced here. The checkpoint and notebook do not establish external clinical validation or clinical utility. The evaluation split is image-level, and patient-level independence cannot be verified from the notebook/data layout. Do not use this prototype as a confirmed diagnosis or substitute for a qualified clinician's judgment.
