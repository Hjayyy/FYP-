# Lightweight NumPy based logistic regression model and startadisation
# utilities used for baseline escalation and prediction experiments
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Numerically stable sigmoid function
def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class NumpyLogisticRegression:
    lr: float = 0.05
    l2: float = 1e-3
    epochs: int = 2000
    class_weights: dict | None = None
    w: np.ndarray | None = None
    b: float = 0.0

    # Train logistic regression using gradient descent
    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        n, d = X.shape
        self.w = np.zeros(d, dtype=float)
        self.b = 0.0

        if self.class_weights is None:
            self.class_weights = {0: 1.0, 1: 1.0}

        for _ in range(self.epochs):
            logits = X @ self.w + self.b
            p = _sigmoid(logits)

            # Apply class weights to reduce the impact of class imbalance
            weights = np.where(
                y == 1,
                self.class_weights.get(1, 1.0),
                self.class_weights.get(0, 1.0),
            )

            error = (p - y) * weights

            grad_w = (X.T @ error) / n + self.l2 * self.w
            grad_b = float(np.mean(error))

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

        return self
    
    # Return predicted probabilities for the positive class
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not trained/loaded.")
        return _sigmoid(X @ self.w + self.b)

    # Return binary predictions using the specified threshold
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # Save the training model to disk
    def save(self, path: str) -> None:
        payload = {
            "w": self.w,
            "b": self.b,
            "lr": self.lr,
            "l2": self.l2,
            "epochs": self.epochs,
            "class_weights": self.class_weights,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    # Load a trained model from disk
    @classmethod
    def load(cls, path: str) -> "NumpyLogisticRegression":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        model = cls(
            lr=float(payload.get("lr", 0.05)),
            l2=float(payload.get("l2", 1e-3)),
            epochs=int(payload.get("epochs", 2000)),
            class_weights=payload.get("class_weights", {0: 1.0, 1: 1.0}),
        )

        model.w = payload["w"]
        model.b = float(payload["b"])
        return model

# Fit feature standardisation statistics
def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma

# Apply feature standardisation using stored mean and standard deviation
def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (X - mu) / sigma


@dataclass
class StandardizedBinaryClassifier:
    
    # Wrapper that combines a binary classifier with feature standardisation
    model: NumpyLogisticRegression
    mu: np.ndarray
    sigma: np.ndarray
    feature_names: List[str]

    # Return predicted probabilities after applying feature standardisation
    def predict_proba_1(self, X: np.ndarray) -> np.ndarray:
        Xs = standardize_apply(X, self.mu, self.sigma)
        return self.model.predict_proba(Xs)

    # Save the wrapped model and its standardisation parameters
    def save(self, path: str) -> None:
        payload = {
            "mu": self.mu,
            "sigma": self.sigma,
            "feature_names": self.feature_names,
            "model_payload": {"w": self.model.w, "b": self.model.b, "lr": self.model.lr, "l2": self.model.l2, "epochs": self.model.epochs},
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

     # Load the wrapped model and its standardisation parameters
    @classmethod
    def load(cls, path: str) -> "StandardizedBinaryClassifier":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        m = NumpyLogisticRegression(
            lr=float(payload["model_payload"].get("lr", 0.05)),
            l2=float(payload["model_payload"].get("l2", 1e-3)),
            epochs=int(payload["model_payload"].get("epochs", 2000)),
        )
        m.w = payload["model_payload"]["w"]
        m.b = float(payload["model_payload"]["b"])
        return cls(model=m, mu=payload["mu"], sigma=payload["sigma"], feature_names=payload["feature_names"])