# Trains escalation prediction models using simulation-generated datasets
# for each organ type and compares baseline and ensemble classifiers.


from __future__ import annotations
import csv
import numpy as np
import joblib
from typing import List, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from ml_model import (
    NumpyLogisticRegression,
    standardize_fit,
    StandardizedBinaryClassifier,
)


# Organ types used for model training.
ORGANS = ["heart", "kidney", "lungs"]

# Organ types used for model training.
FEATURE_NAMES = [
    "temperature_c",
    "elapsed_minutes",
    "delay_minutes",
    "delayed_this_minute",
    "distance_remaining_km",
    "anomaly_score",
    "remaining_safe_minutes",
]

RESULTS_FILE = "model_comparison_results.csv"


# Load a labelled dataset from CSV
def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    X: List[List[float]] = []
    y: List[int] = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            X.append([float(r[name]) for name in FEATURE_NAMES])
            y.append(int(r["label_escalate"]))

    return np.array(X, dtype=float), np.array(y, dtype=float)


# Split the dataset into training and validation subsets.
def train_val_split(X, y, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    n_val = int(len(X) * val_frac)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


# Compute classification metrics from predictions
def metrics(y_true, y_pred):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    acc = (tp + tn) / max(1, len(y_true))

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# Train and evaluate models for a single organ type
def train_for_organ(organ: str, threshold=0.5):

    print("\n" + "=" * 60)
    print(f"TRAINING MODELS FOR: {organ.upper()}")
    print("=" * 60)

    dataset_path = f"dataset_{organ}.csv"
    X, y = load_dataset(dataset_path)
    X_tr, y_tr, X_val, y_val = train_val_split(X, y)

    
    # Model 1: Logistic Regression baseline
    X_tr_s, mu, sigma = standardize_fit(X_tr)
    X_val_s = (X_val - mu) / np.where(sigma < 1e-8, 1.0, sigma)

    # Compute class weights to reduce the impact of class imbalance
    n_total = len(y_tr)
    n_pos = np.sum(y_tr == 1)
    n_neg = np.sum(y_tr == 0)

    weight_pos = n_total / (2 * n_pos)
    weight_neg = n_total / (2 * n_neg)

    class_weights = {0: weight_neg, 1: weight_pos}

    log_model = NumpyLogisticRegression(
        lr=0.05,
        l2=1e-4,
        epochs=2500,
        class_weights=class_weights
    ).fit(X_tr_s, y_tr)

    log_wrapped = StandardizedBinaryClassifier(
        model=log_model,
        mu=mu,
        sigma=sigma,
        feature_names=FEATURE_NAMES,
    )

    log_wrapped.save(f"{organ}_logreg.pkl") # Save the trained logistic regression model

    p_log = log_model.predict_proba(X_val_s)
    y_hat_log = (p_log >= threshold).astype(int)
    m_log = metrics(y_val, y_hat_log)

    
    # Model 2: Gradient Boosting classifier
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    gb_model.fit(X_tr, y_tr)
    p_gb = gb_model.predict_proba(X_val)[:, 1]
    y_hat_gb = (p_gb >= threshold).astype(int)
    m_gb = metrics(y_val, y_hat_gb)

    joblib.dump(gb_model, f"{organ}_gboost.pkl") # Save the trained Gradient Boosting model

    # print results
    print("\nLogistic Regression Metrics:")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in m_log.items()})

    print("\nGradient Boosting Metrics:")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in m_gb.items()})

    # saves data to csv 
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            organ, "logreg",
            m_log["acc"], m_log["precision"],
            m_log["recall"], m_log["f1"]
        ])
        writer.writerow([
            organ, "gboost",
            m_gb["acc"], m_gb["precision"],
            m_gb["recall"], m_gb["f1"]
        ])


if __name__ == "__main__":

    # Reset the results file and train models for all organs.
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["organ", "model", "accuracy", "precision", "recall", "f1"])

    for organ in ORGANS:
        train_for_organ(organ)