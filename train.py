"""
train.py — Build feature matrix, compare classifiers, train best, save model.

Tries: Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting.
Picks the classifier with the highest LOO-CV accuracy.

Usage:
    python train.py

Data directories (edit if needed):
    REAL_DIR   = real photos (label 0)
    SCREEN_DIR = screen recaptures (label 1)
"""

import os
import json
import glob
import numpy as np

from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import accuracy_score, classification_report, confusion_matrix

from features import extract_features, FEATURE_NAMES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REAL_DIR   = r"C:\Users\hp\Downloads\real_img"
SCREEN_DIR = r"C:\Users\hp\Downloads\screen_img"
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.json")

IMAGE_EXTENSIONS = ("*.jpeg", "*.jpg", "*.png", "*.bmp", "*.webp")


def collect_images(folder: str):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
        paths.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(set(paths))


def build_dataset():
    real_imgs   = collect_images(REAL_DIR)
    screen_imgs = collect_images(SCREEN_DIR)
    print(f"Found {len(real_imgs)} real, {len(screen_imgs)} screen images.")
    print("Extracting features (may take 1-2 minutes)...\n")

    X, y = [], []
    for i, path in enumerate(real_imgs):
        try:
            X.append(extract_features(path))
            y.append(0)
            print(f"  [real   {i+1:>3}/{len(real_imgs)}] {os.path.basename(path)}")
        except Exception as e:
            print(f"  [SKIP] {path}: {e}")

    for i, path in enumerate(screen_imgs):
        try:
            X.append(extract_features(path))
            y.append(1)
            print(f"  [screen {i+1:>3}/{len(screen_imgs)}] {os.path.basename(path)}")
        except Exception as e:
            print(f"  [SKIP] {path}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train():
    X, y = build_dataset()
    print(f"\nDataset: {X.shape[0]} images x {X.shape[1]} features")
    print(f"  Real (0): {(y==0).sum()}   Screen (1): {(y==1).sum()}\n")

    # Scale features (fit on all data for the final model)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Compare classifiers with LOO-CV ──────────────────────────────────────
    candidates = {
        "LogisticRegression":   LogisticRegression(C=1.0, max_iter=2000, random_state=42),
        "RandomForest_200":     RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "RandomForest_300":     RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42),
        "ExtraTrees_200":       ExtraTreesClassifier(n_estimators=200, max_depth=7, random_state=42),
        "ExtraTrees_300":       ExtraTreesClassifier(n_estimators=300, max_depth=None, random_state=42),
        "GradientBoosting":     GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        "HistGBM_lr005":        HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=31, random_state=42),
        "HistGBM_lr01":         HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, random_state=42),
        "HistGBM_lr001":        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.01, max_leaf_nodes=63, random_state=42),
    }

    print("Comparing classifiers via LOO-CV...\n")
    loo = LeaveOneOut()
    results = {}
    for name, clf in candidates.items():
        scores = cross_val_score(clf, X_scaled, y, cv=loo, scoring="accuracy")
        acc = scores.mean()
        results[name] = acc
        print(f"  {name:<28} LOO-CV: {acc*100:.1f}%")

    best_name = max(results, key=results.get)
    best_acc  = results[best_name]
    print(f"\n  Best: {best_name} at {best_acc*100:.1f}%\n")

    # ── Train best classifier on full dataset ─────────────────────────────────
    best_clf = candidates[best_name]
    best_clf.fit(X_scaled, y)

    # ── Full LOO confusion matrix for the best classifier ────────────────────
    y_pred_loo = []
    y_prob_loo = []
    for train_idx, test_idx in loo.split(X_scaled):
        clf_fold = candidates[best_name].__class__(**candidates[best_name].get_params())
        clf_fold.fit(X_scaled[train_idx], y[train_idx])
        prob = clf_fold.predict_proba(X_scaled[test_idx])[0, 1]
        y_pred_loo.append(int(prob >= 0.5))
        y_prob_loo.append(prob)

    y_pred_loo = np.array(y_pred_loo)

    print("=" * 56)
    print(f"  Final LOO-CV Accuracy : {best_acc*100:.1f}%  [{best_name}]")
    print("=" * 56)
    print(classification_report(y, y_pred_loo, target_names=["real (0)", "screen (1)"]))
    cm = confusion_matrix(y, y_pred_loo)
    print("Confusion Matrix:")
    print(cm)

    # ── Feature coefficients (for logistic regression only) ───────────────────
    if hasattr(best_clf, "coef_"):
        print("\nFeature coefficients:")
        for name, coef in sorted(zip(FEATURE_NAMES, best_clf.coef_[0]), key=lambda x: -abs(x[1])):
            bar  = "#" * min(int(abs(coef) * 5), 20)
            sign = "+" if coef > 0 else "-"
            print(f"  {sign}  {bar:<20}  {coef:+.4f}  {name}")
    elif hasattr(best_clf, "feature_importances_"):
        print("\nFeature importances (Random Forest / GB):")
        for name, imp in sorted(zip(FEATURE_NAMES, best_clf.feature_importances_), key=lambda x: -x[1]):
            bar = "#" * int(imp * 100)
            print(f"  {bar:<20}  {imp:.4f}  {name}")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Determine the type so predict.py knows how to load it
    clf_type = best_name.split("_")[0]  # "LogisticRegression", "SVM", "RandomForest", etc.

    if "LogisticRegression" in best_name:
        model_data = {
            "clf_type":       "logistic",
            "clf_name":       best_name,
            "feature_names":  FEATURE_NAMES,
            "scaler_mean":    scaler.mean_.tolist(),
            "scaler_scale":   scaler.scale_.tolist(),
            "coef":           best_clf.coef_[0].tolist(),
            "intercept":      float(best_clf.intercept_[0]),
            "loo_accuracy":   round(best_acc, 4),
            "n_total": int(len(y)), "n_real": int((y==0).sum()), "n_screen": int((y==1).sum()),
        }
    elif "SVM" in best_name:
        # Save SVM support vectors and weights
        model_data = {
            "clf_type":        "svm",
            "clf_name":        best_name,
            "feature_names":   FEATURE_NAMES,
            "scaler_mean":     scaler.mean_.tolist(),
            "scaler_scale":    scaler.scale_.tolist(),
            "support_vectors": best_clf.support_vectors_.tolist(),
            "dual_coef":       best_clf.dual_coef_.tolist(),
            "intercept":       best_clf.intercept_.tolist(),
            "gamma":           float(best_clf._gamma),
            "kernel":          "rbf",
            "loo_accuracy":    round(best_acc, 4),
            "n_total": int(len(y)), "n_real": int((y==0).sum()), "n_screen": int((y==1).sum()),
        }
    else:
        # For tree-based models, we save importances but use sklearn's predict_proba
        # We'll pickle just the pipeline for these
        import pickle
        pipeline = Pipeline([("scaler", scaler), ("clf", best_clf)])
        pkl_path = os.path.join(MODEL_DIR, "classifier.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(pipeline, f)
        model_data = {
            "clf_type":       "pkl",
            "clf_name":       best_name,
            "pkl_path":       pkl_path,
            "feature_names":  FEATURE_NAMES,
            "scaler_mean":    scaler.mean_.tolist(),
            "scaler_scale":   scaler.scale_.tolist(),
            "loo_accuracy":   round(best_acc, 4),
            "n_total": int(len(y)), "n_real": int((y==0).sum()), "n_screen": int((y==1).sum()),
        }

    with open(MODEL_PATH, "w") as f:
        json.dump(model_data, f, indent=2)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"  Classifier : {best_name}")
    print(f"  LOO-CV Acc : {best_acc*100:.1f}%")
    print("\nAll classifier results:")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " <-- BEST" if name == best_name else ""
        print(f"  {name:<28} {acc*100:.1f}%{marker}")

    print("\nDone! Run predict.py or evaluate.py next.")


if __name__ == "__main__":
    train()
