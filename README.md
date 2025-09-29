## Skin Disease Classifier Prototype

### 1) Prepare dataset splits

```bash
python dataset_split.py
```

This creates `dataset/{train,val,test}/<class>` from `skin_disease_dataset` without leakage.

### 2) Train the model

```bash
python train.py
```

Artifacts:
- `best_model.keras` (best validation accuracy)
- `skin_classifier_model.keras` (final saved model)

### 3) Run API

```bash
uvicorn api:APP --host 0.0.0.0 --port 8000
```

POST `/predict` with form file `file` containing an image. Returns class probabilities.

### Notes
- Uses MobileNetV2 with fine-tuning, on-the-fly augmentation, and early stopping.
- Aim: ≥70% accuracy on held-out test set. Results depend on data quality.

