"""
predict.py — Inference entrypoint for the photo authenticity detector.

Contract:
    predict(image_path: str) -> float   (score in [0, 1])
    Score -> 0 means "real photo", score -> 1 means "screenshoted / recaptured".

The model weights are loaded ONCE at import time.
No GPU, no network call, no heavy framework.

Usage:
    from predict import predict
    score = predict("path/to/image.jpg")
    print(f"Screen probability: {score:.3f}")

    # Or from command line:
    python predict.py path/to/image.jpg
"""

import sys
import json
import os
import pickle
import numpy as np

from features import extract_features

# ---------------------------------------------------------------------------
# Load model weights once at module import (fast)
# ---------------------------------------------------------------------------
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_MODEL_DIR, "model", "classifier.json")

def _load_model(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. Please run train.py first."
        )
    with open(path) as f:
        return json.load(f)

_MODEL = _load_model(_MODEL_PATH)
_CLF_TYPE = _MODEL["clf_type"]

_SCALER_MEAN  = np.array(_MODEL["scaler_mean"],  dtype=np.float64)
_SCALER_SCALE = np.array(_MODEL["scaler_scale"], dtype=np.float64)

# Load classifier specifics based on type
_PICKLED_PIPELINE = None
_COEF = None
_INTERCEPT = None

if _CLF_TYPE == "logistic":
    _COEF = np.array(_MODEL["coef"], dtype=np.float64)
    _INTERCEPT = float(_MODEL["intercept"])
elif _CLF_TYPE == "svm":
    # Reconstruct SVM decision function or use sklearn.
    # To keep dependencies light but avoid writing full SVM inference in pure python:
    # We can reconstruct it or simply use a fallback. Actually, to keep it extremely reliable
    # and fast, we can import SVC and reconstruct, or we can use a pickled pipeline if needed.
    # Wait, train.py saves SVM parameters in JSON. Let's reconstruct RBF SVM inference:
    # f(x) = sum(dual_coef_i * K(sv_i, x)) + intercept
    # where K(u, v) = exp(-gamma * ||u - v||^2)
    _SUPPORT_VECTORS = np.array(_MODEL["support_vectors"], dtype=np.float64)
    _DUAL_COEF = np.array(_MODEL["dual_coef"], dtype=np.float64)[0] # dual_coef is shape (1, n_sv)
    _INTERCEPT = float(_MODEL["intercept"][0])
    _GAMMA = float(_MODEL["gamma"])
elif _CLF_TYPE == "pkl":
    # Fallback to loading the pickled sklearn pipeline directly (Random Forest / GB)
    pkl_file = os.path.join(_MODEL_DIR, "model", "classifier.pkl")
    with open(pkl_file, "rb") as f:
        _PICKLED_PIPELINE = pickle.load(f)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


def predict(image_path: str) -> float:
    """
    Predict whether a photo is real or a recapture of a screen.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.

    Returns
    -------
    float in [0, 1]
        Close to 0 -> real photo.
        Close to 1 -> photo taken of a screen (fake / recaptured).
    """
    # 1. Extract the 10-dimensional feature vector
    features = extract_features(image_path).astype(np.float64)

    # 2. Standardise and predict depending on classifier type
    if _CLF_TYPE == "pkl":
        # Let the pickled pipeline handle scaling and prediction
        # predict_proba returns [prob_class_0, prob_class_1]
        prob = _PICKLED_PIPELINE.predict_proba(features.reshape(1, -1))[0, 1]
        return float(prob)
    
    # Otherwise, scale manually
    features_scaled = (features - _SCALER_MEAN) / (_SCALER_SCALE + 1e-8)

    if _CLF_TYPE == "logistic":
        logit = float(np.dot(_COEF, features_scaled)) + _INTERCEPT
        score = _sigmoid(logit)
        return score

    elif _CLF_TYPE == "svm":
        # SVM RBF decision function: f(x) = sum(dual_coef_i * exp(-gamma * ||sv_i - x||^2)) + intercept
        # Since SVC probability=True uses Platt scaling, let's compute decision_function
        # and Platt scale it (or just sigmoid-map it if Platt parameters aren't fully saved).
        # To be simple and robust: SVC probability maps decision value to probability.
        # Let's compute decision_value:
        diffs = _SUPPORT_VECTORS - features_scaled
        sq_dists = np.sum(diffs ** 2, axis=1)
        k = np.exp(-_GAMMA * sq_dists)
        decision_value = float(np.dot(_DUAL_COEF, k)) + _INTERCEPT
        # Map decision value to [0, 1] using standard sigmoid as Platt fallback
        score = _sigmoid(decision_value)
        return score

    return 0.5


# ---------------------------------------------------------------------------
# CLI usage: python predict.py <image_path>
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    score = predict(path)

    label = "SCREEN (fake)" if score >= 0.5 else "REAL"
    confidence = max(score, 1 - score) * 100
    print(f"\nImage : {path}")
    print(f"Score : {score:.4f}  ->  {label}  (confidence {confidence:.1f}%)")
    print()
    print("  0.0 ------------------------------ 1.0")
    marker_pos = int(score * 40)
    bar = " " * marker_pos + "^"
    print(f"  {bar}")
    print("  REAL                          SCREEN")
