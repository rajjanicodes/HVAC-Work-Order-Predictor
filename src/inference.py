"""Model loading and inference."""

import numpy as np
import joblib
from datetime import datetime, timezone

from src.config import ARTIFACTS_DIR, LABEL_MAPPING
from src.features import extract_features, FEATURE_NAMES

# Load model and scaler once - eliminates per-request I/O overhead
model = joblib.load(ARTIFACTS_DIR / "logistic.joblib")
scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
scaler.feature_names_in_ = None  # scaler was fitted on a DataFrame; clear names so numpy arrays pass cleanly

# Index of the WO class (1) inside model.classes_ - used to pull out prob(WO) from predict_proba
wo_idx = list(model.classes_).index(1)


def predict(records: list[dict]) -> list[dict]:
    """Takes a list of records (each with 'timeseries_data'), returns predictions."""

    # Extract 23 statistical features from each record, then scale
    rows = [extract_features(r["timeseries_data"]) for r in records]
    X = np.array([[row[name] for name in FEATURE_NAMES] for row in rows])
    X = scaler.transform(X)

    # Single model call for the whole batch
    preds = model.predict(X)
    probas = model.predict_proba(X)
    now = datetime.now(timezone.utc)

    return [
        {
            "label":        LABEL_MAPPING[str(p)],
            "confidence":   float(probas[i][int(p)]),   # probability of the winning class
            "prob_wo":      float(probas[i][wo_idx]),    # probability of WO - drives risk_level
            "predicted_at": now,
        }
        for i, p in enumerate(preds)
    ]
