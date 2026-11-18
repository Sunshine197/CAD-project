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

The script reproduces the notebook's stratified 60/20/20 image-level split (`random_state=42`), fits one temperature-scaling parameter on the validation split, and reports aggregate classification and calibration metrics on validation and test data. It excludes per-image scores and image paths from the JSON output. This is intended as an evaluation harness; it cannot certify clinical validity.

## Validation performed

The supplied checkpoint loaded successfully with strict state-dictionary matching using the reconstructed architecture. A smoke test on a synthetic blank image confirmed that inference returns all three classes and scores summing to approximately 1.0. That synthetic-image output is **not** a medical prediction and is not used as performance evidence.

## Important limitations

The softmax scores are not established as calibrated probabilities until evaluation has been run and reviewed. The original notebook reports 88.53% test accuracy on its own held-out test split; this has not been independently reproduced here. The checkpoint and notebook do not establish external clinical validation or clinical utility. The evaluation split is image-level, and patient-level independence cannot be verified from the notebook/data layout. Do not use this prototype as a confirmed diagnosis or substitute for a qualified clinician's judgment.
