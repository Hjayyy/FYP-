# Core intelligence
# Evaluate transport risk using temperature thresholds, transport duration,
# delay severity, and anomaly detection.

from datetime import datetime, timezone
from models import RiskState, TemperatureReading, TransportState
from organ_profile import ORGAN_PROFILES
from anomaly_detector import EWMAAnomalyDetector


# Initialise the risk engine for a specific organ type
class RiskEngine:
    def __init__(self, organ_key: str):
        # stores the organ being transported and loads its constraints
        self.organ_key = organ_key 
        self.profile = ORGAN_PROFILES[organ_key]

      
        self.temp_anomaly = EWMAAnomalyDetector(mean=4.0, var=0.5)

    # Map the numerical risk score to a categorical risk level
    def _risk_level_from_score(self, score: float) -> str:
        if score < 1:
            return "LOW"
        if score < 2.0:
            return "MODERATE"
        if score < 3:
            return "HIGH"
        return "CRITICAL"

    # Evaluate the current transport state and compute the overall risk.
    def evaluate(self, temp: TemperatureReading, transport: TransportState) -> RiskState:
        min_t, max_t = self.profile.safe_temp_range
        max_time = self.profile.max_transport_minutes

        risk_score = 0.0
        reasons = []

        # Assess whether temperature is outside or near the safe storage range
        out_of_range = temp.temperature_c < min_t or temp.temperature_c > max_t
        near_upper = temp.temperature_c > (max_t - 0.5)
        near_lower = temp.temperature_c < (min_t + 0.2)

        # Rule based intelligence
        if out_of_range:
            risk_score += 2.0
            reasons.append(f"Temperature out of safe range ({min_t}-{max_t}Â°C)")
        elif near_upper:
            risk_score += 1.0
            reasons.append(f"Temperature near upper safe limit ({max_t}Â°C)")
        elif near_lower:
            risk_score += 0.5
            reasons.append(f"Temperature near lower safe limit ({min_t}Â°C)")

        # Increase risk as the journey approaches or exceeds the allowed time window.
        total_time = transport.elapsed_minutes
        remaining = max(0, max_time - total_time)

        if total_time > max_time:
            risk_score += 2.0
            reasons.append(f"Transport time exceeded max ({max_time} min)")
        elif total_time > 0.8 * max_time:
            risk_score += 1.0
            reasons.append("Transport time exceeds 80% of allowed window")

        # Increase risk when transport delays accumulate.
        if transport.delay_minutes > 60:
            risk_score += 1.5
            reasons.append("Severe delays (>60 minutes)")

        elif transport.delay_minutes > 30:
            risk_score += 1.0
            reasons.append("Moderate delays (>30 minutes)")

        # EWMA anomaly detection identifies unusual temperature behaviour
        # that may indicate cooling instability or environmental disruption
        ewma_anom = self.temp_anomaly.update(temp.temperature_c)
        anomaly = ewma_anom

        # Adjust anomaly contribution using additional transport context
        if out_of_range:
            anomaly = max(anomaly, 0.6)
        if transport.delayed_this_minute:
            anomaly = max(anomaly, 0.4)
        if transport.delay_minutes > 60:
            anomaly = max(anomaly, 0.7)

        risk_score += anomaly

        # Estimate confidence in the current risk classification.
        confidence = 0.4 + min(0.5, 0.12 * len(reasons)) + min(0.1, anomaly * 0.1)
        confidence = max(0.4, min(1.0, confidence))


        level = self._risk_level_from_score(risk_score)

        # Return the evaluated risk state.
        return RiskState(
            timestamp=datetime.now(timezone.utc),
            organ_key=self.organ_key,
            risk_score=round(risk_score, 2),
            risk_level=level,
            requires_intervention=level in ("HIGH", "CRITICAL"),
            anomaly_score=round(anomaly, 2),
            confidence=round(confidence, 2),
            remaining_safe_minutes=int(remaining),
            reasons=reasons
        )
