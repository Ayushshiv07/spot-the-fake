"""
evaluate.py — Full evaluation report: LOOCV, confusion matrix, ROC-AUC,
              and recommended operating threshold.

Run AFTER train.py has saved model/classifier.json.

Usage:
    python evaluate.py
"""

import os
import json
import glob
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)

from features import extract_features, FEATURE_NAMES

REAL_DIR   = r"C:\Users\hp\Downloads\real_img"
SCREEN_DIR = r"C:\Users\hp\Downloads\screen_img"
IMAGE_EXTENSIONS = ("*.jpeg", "*.jpg", "*.png", "*.bmp", "*.webp")


def collect_images(folder):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
        paths.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(set(paths))


def build_dataset():
    real_imgs   = collect_images(REAL_DIR)
    screen_imgs = collect_images(SCREEN_DIR)
    print(f"Real: {len(real_imgs)}  Screen: {len(screen_imgs)}")
    X, y = [], []
    for path in real_imgs:
        try:
            X.append(extract_features(path))
            y.append(0)
        except Exception as e:
            print(f"  [SKIP] {path}: {e}")
    for path in screen_imgs:
        try:
            X.append(extract_features(path))
            y.append(1)
        except Exception as e:
            print(f"  [SKIP] {path}: {e}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def evaluate():
    print("Loading and extracting features (may take 1-2 minutes)...")
    X, y = build_dataset()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Load model information to see which one was trained
    model_path = os.path.join(os.path.dirname(__file__), "model", "classifier.json")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'. Running evaluation with LogisticRegression defaults.")
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
        clf_name = "LogisticRegression"
    else:
        with open(model_path) as f:
            model_info = json.load(f)
        clf_name = model_info["clf_name"]
        
        # Instantiate candidate corresponding to saved name
        candidates = {
            "LogisticRegression": LogisticRegression(C=1.0, max_iter=2000, random_state=42),
            "LogisticRegression_C5": LogisticRegression(C=5.0, max_iter=2000, random_state=42),
            "SVM_RBF":           SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42),
            "SVM_RBF_C50":       SVC(kernel="rbf", C=50.0, gamma="scale", probability=True, random_state=42),
            "RandomForest":      RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
            "GradientBoosting":  GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        }
        clf = candidates.get(clf_name, LogisticRegression(C=1.0, max_iter=2000, random_state=42))

    # ── Leave-One-Out CV ─────────────────────────────────────────────────────
    print(f"\nRunning Leave-One-Out CV for classifier: {clf_name}...")
    loo = LeaveOneOut()
    y_true_loo, y_pred_loo, y_prob_loo = [], [], []

    for train_idx, test_idx in loo.split(X_scaled):
        # Create a fresh clone/instance of the same classifier configuration
        clf_fold = clf.__class__(**clf.get_params())
        clf_fold.fit(X_scaled[train_idx], y[train_idx])
        prob = clf_fold.predict_proba(X_scaled[test_idx])[0, 1]
        pred = int(prob >= 0.5)
        y_true_loo.append(y[test_idx[0]])
        y_pred_loo.append(pred)
        y_prob_loo.append(prob)

    y_true_loo = np.array(y_true_loo)
    y_pred_loo = np.array(y_pred_loo)
    y_prob_loo = np.array(y_prob_loo)

    loo_acc = accuracy_score(y_true_loo, y_pred_loo)
    roc_auc = roc_auc_score(y_true_loo, y_prob_loo)

    print("\n" + "=" * 56)
    print(f"  Leave-One-Out Accuracy : {loo_acc*100:.1f}%")
    print(f"  ROC-AUC                : {roc_auc:.4f}")
    print("=" * 56)

    print("\nClassification Report (LOO-CV):")
    print(classification_report(y_true_loo, y_pred_loo,
                                 target_names=["real (0)", "screen (1)"]))

    cm = confusion_matrix(y_true_loo, y_pred_loo)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Real  Screen")
    print(f"Actual Real   {tn:>4}   {fp:>4}   (FP = {fp} real photos flagged as fake)")
    print(f"Actual Screen {fn:>4}   {tp:>4}   (FN = {fn} fakes missed)")

    # ── Threshold recommendation ─────────────────────────────────────────────
    print("\nROC Curve — threshold analysis:")
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true_loo, y_prob_loo)

    # Find threshold that maximises F1 (balanced)
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.2, 0.8, 0.01):
        preds = (y_prob_loo >= thresh).astype(int)
        tp_ = np.sum((preds == 1) & (y_true_loo == 1))
        fp_ = np.sum((preds == 1) & (y_true_loo == 0))
        fn_ = np.sum((preds == 0) & (y_true_loo == 1))
        prec = tp_ / (tp_ + fp_ + 1e-8)
        rec  = tp_ / (tp_ + fn_ + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    # Threshold that favours precision on real class (fewer false positives)
    print(f"  Best balanced threshold (max F1)    : {best_thresh:.2f}  (F1={best_f1:.3f})")
    print(f"  Default threshold                    : 0.50")
    print(f"\n  Business tradeoff:")
    print(f"    • Flagging a real photo as fake angers genuine users.")
    print(f"    • Use threshold > 0.5 to reduce false positives on real class.")
    print(f"    • Use threshold < 0.5 to catch more fakes (higher recall on screen).")

    # ── Feature importance ───────────────────────────────────────────────────
    clf_all = clf.__class__(**clf.get_params())
    clf_all.fit(X_scaled, y)
    
    if hasattr(clf_all, "coef_"):
        print("\nFeature Importance (full-data coefficients):")
        coefs = clf_all.coef_[0]
        for name, coef in sorted(zip(FEATURE_NAMES, coefs), key=lambda x: -abs(x[1])):
            bar = "#" * min(int(abs(coef) * 6), 30)
            sign = "->screen" if coef > 0 else "->real  "
            print(f"  {sign}  {bar:<30}  {coef:+.4f}  {name}")
    elif hasattr(clf_all, "feature_importances_"):
        print("\nFeature Importances (full-data importances):")
        importances = clf_all.feature_importances_
        for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
            bar = "#" * int(imp * 100)
            print(f"  {bar:<30}  {imp:.4f}  {name}")

    print(f"\n[OK] Honest LOO-CV accuracy: {loo_acc*100:.1f}%  |  ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    evaluate()
