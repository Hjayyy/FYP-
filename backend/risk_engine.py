# core intelligence!!!
# combining the temperature logic with the delay logic
# output of risk classification: low, medium and high
# this place wil trigger alerts, will justify ml integration later

from datetime import datetime, timezone

from organ_profile import ORGAN_PROFILES, OrganProfile


class RiskEngine:
    def __init__(self, organ_type: str):
        # Initialise risk engine with an organ-specific profile.
        self.organ_type = organ_type
        self.profile: OrganProfile = ORGAN_PROFILES[organ_type]

    def evaluate(self, temperature_data: dict, transport_data: dict) -> dict:
        # Evaluates risk based on: Temperature deviation, Transport duration and delays
        # Returns a structured risk state used for alerts and rerouting decisions.

        temp = temperature_data["temperature"]
        elapsed = transport_data["elapsed_minutes"]
        delays = transport_data["delay_minutes"]

        min_temp, max_temp = self.profile.safe_temp_range
        max_time = self.profile.max_transport_minutes

        risk_score = 0.0

        # 1. Temperature-based risk
        if temp < min_temp or temp > max_temp:
            # Outside safe storage range
            risk_score += 2.0
        elif temp > (max_temp - 1.0):
            # Approaching unsafe threshold
            risk_score += 1.0

        # 2. Time-based ischemic risk

        total_time = elapsed + delays

        if total_time > max_time:
            risk_score += 2.0
        elif total_time > 0.8 * max_time:
            risk_score += 1.0


        # 3. Delay amplification

        if delays > 30:
            risk_score += 1.0


        # Risk classification

        if risk_score >= 4.0:
            risk_level = "CRITICAL"
        elif risk_score >= 2.5:
            risk_level = "HIGH"
        elif risk_score >= 1.0:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "organ": self.organ_type,
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "requires_intervention": risk_level in ["HIGH", "CRITICAL"]
        }
