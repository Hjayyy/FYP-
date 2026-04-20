# Construct the feature dictionary used by the escalation prediction model.

from __future__ import annotations
from typing import Dict
from models import OrganProfile, TemperatureReading, TransportState, RiskState


# Convert simulation state into ML feature inputs
def make_feature_dict(
    organ: OrganProfile,
    temp: TemperatureReading,
    transport: TransportState,
    risk: RiskState,
) -> Dict[str, float]:
    
    # Centralised feature definition to ensure consistency between
    # training and inference.
    return {

        # Current organ temperature
        "temperature_c": float(temp.temperature_c),
        # Total elapsed transport time
        "elapsed_minutes": float(transport.elapsed_minutes),
        # traffic delay minutes
        "delay_minutes": float(transport.delay_minutes),
        # Binary indicator for congestion in this timestep
        "delayed_this_minute": 1.0 if transport.delayed_this_minute else 0.0,
        "distance_remaining_km": float(transport.distance_remaining_km),
        # Temperature anomaly score from EWMA detector
        "anomaly_score": float(risk.anomaly_score),
        # Remaining safe viability window for the organ
        "remaining_safe_minutes": float(risk.remaining_safe_minutes),
    }