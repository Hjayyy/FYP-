# Machine learning predictor used to estimate near-term escalation probability.
# Acts as a predictive layer on top of the rule-based risk engine

from __future__ import annotations
import numpy as np
import joblib

from ml_features import make_feature_dict
from models import OrganProfile, TemperatureReading, TransportState, RiskState


class MLRiskPredictor:

   # Load the trained model and define the expected feature order
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

        # The model is stored as a serialized sklearn object (.pkl)
        # and loaded using joblib

        self.feature_names = [
            "temperature_c",
            "elapsed_minutes",
            "delay_minutes",
            "delayed_this_minute",
            "distance_remaining_km",
            "anomaly_score",
            "remaining_safe_minutes",
        ]
    # Construct the feature vector from the current simulation state and
    # return the probability of escalation
    def predict_escalation_prob(
        self,
        organ: OrganProfile,
        temp: TemperatureReading,
        transport: TransportState,
        risk: RiskState,
    ) -> float:

        # Convert the current simulation state into the feature representation
        # expected by the trained model
        feat = make_feature_dict(organ, temp, transport, risk)

        # Arrange features in the exact order expected by the model
        try:
            X = np.array([[feat[name] for name in self.feature_names]])
        except KeyError as e:
            raise KeyError(
                f"Missing feature {e}. Required features: {self.feature_names}"
            )
        # Predict probability that the current state will escalate
        prob = float(self.model.predict_proba(X)[0][1])

        # Ensure the returned value remains a valid probability
        if prob < 0:
            prob = 0.0
        if prob > 1:
            prob = 1.0

        return prob