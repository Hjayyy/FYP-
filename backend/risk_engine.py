# core intelligence!!!
# combining the temperature logic with the delay logic
# output of risk classification: low, medium and high
# this place wil trigger alerts, will justify ml integration later

from datetime import datetime, timezone
from models import RiskState, TemperatureReading, TransportState
from organ_profile import ORGAN_PROFILES
from anomaly_detector import EWMAAnomalyDetector


class RiskEngine:
    def __init__(self, organ_key: str):
        self.organ_key = organ_key
        self.profile = ORGAN_PROFILES[organ_key]

        # Now anomaly_detector has a purpose (online signal)
        self.temp_anomaly = EWMAAnomalyDetector(mean=4.0, var=0.5)

    def _risk_level_from_score(self, score: float) -> str:
        if score < 1.5:
            return "LOW"
        if score < 3.0:
            return "MODERATE"
        if score < 4.5:
            return "HIGH"
        return "CRITICAL"

    def evaluate(self, temp: TemperatureReading, transport: TransportState) -> RiskState:
        min_t, max_t = self.profile.safe_temp_range
        max_time = self.profile.max_transport_minutes

        risk_score = 0.0
        reasons = []

        # -------- Temperature risk --------
        out_of_range = temp.temperature_c < min_t or temp.temperature_c > max_t
        near_upper = temp.temperature_c > (max_t - 1.0)
        near_lower = temp.temperature_c < (min_t + 1.0)

        if out_of_range:
            risk_score += 2.0
            reasons.append(f"Temperature out of safe range ({min_t}-{max_t}°C)")
        elif near_upper:
            risk_score += 1.0
            reasons.append(f"Temperature near upper safe limit ({max_t}°C)")
        elif near_lower:
            risk_score += 0.5
            reasons.append(f"Temperature near lower safe limit ({min_t}°C)")

        # -------- Time risk --------
        # elapsed_minutes already includes delay minutes (do NOT add delay again)
        total_time = transport.elapsed_minutes
        remaining = max(0, max_time - total_time)

        if total_time > max_time:
            risk_score += 2.0
            reasons.append(f"Transport time exceeded max ({max_time} min)")
        elif total_time > 0.8 * max_time:
            risk_score += 1.0
            reasons.append("Transport time exceeds 80% of allowed window")

        # -------- Delay amplification --------
        if transport.delay_minutes > 30:
            risk_score += 1.0
            reasons.append("Delays exceed 30 minutes")

        # -------- Anomaly score (EWMA + context) --------
        ewma_anom = self.temp_anomaly.update(temp.temperature_c)
        anomaly = ewma_anom

        # add context bumps (keeps it interpretable)
        if out_of_range:
            anomaly = max(anomaly, 0.6)
        if transport.delayed_this_minute:
            anomaly = max(anomaly, 0.4)
        if transport.delay_minutes > 60:
            anomaly = max(anomaly, 0.7)

        # -------- Confidence --------
        # simple: reasons increase confidence + strong anomaly increases confidence
        confidence = 0.4 + min(0.5, 0.12 * len(reasons)) + min(0.1, anomaly * 0.1)
        confidence = max(0.4, min(1.0, confidence))

        level = self._risk_level_from_score(risk_score)

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
