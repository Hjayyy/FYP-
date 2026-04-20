# Select the safest destination hospital under elevated transport risk
# using a multi-factor routing cost model.

from __future__ import annotations

from typing import Dict, List, Tuple, Union, Optional
from models import Hospital, RiskState, TransportState, Coord, OrganProfile
from transport_simulator import haversine
from road_network import risk_aware_shortest_path

# Numerical ordering of risk levels used for scaling penalties 
RISK_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}

# Risk multiplier increases the impact of routing penalties
# as the organ transport becomes more critical.

def _risk_multiplier(level: str) -> float:
    if level == "MODERATE":
        return 1.5
    if level == "HIGH":
        return 2.5
    if level == "CRITICAL":
        return 4.0
    return 1.0

# Hospitals with higher clinical capability reduce the overall routing cost,
# especially under higher-risk conditions.
def _capability_bonus(h: Hospital, risk: RiskState) -> float:

    cap = float(getattr(h, "capability", 0.70))  # default if missing
    scale = 2.0 + 6.0 * (RISK_ORDER[risk.risk_level] / 3.0)  # LOW->2, CRIT->8
    return cap * scale


def _expected_wait_penalty(h: Hospital, risk: RiskState, organ: OrganProfile) -> float:
    
    # Waiting time becomes more costly as the remaining viability window decreases.
    wait_min = float(getattr(h, "expected_wait_min", 0.0))
    urgency = 1.0 - min(1.0, risk.remaining_safe_minutes / max(1, organ.max_transport_minutes))
    return wait_min * (1.0 + 4.0 * urgency)

# Compute the routing cost breakdown for a candidate hospital.
# This combines logistical and clinical factors.
def cost_breakdown(
    current: Coord,
    h: Hospital,
    risk: RiskState,
    organ: OrganProfile,
    transport: TransportState,
) -> Dict[str, float]:
    
    # This breakdown is returned for transparency and explainability in the monitoring dashboard.
    distance_km = haversine(current, h.location) # calculates the geo distance bwteen the vehicle and the hospital
    rm = _risk_multiplier(risk.risk_level)
    traffic_penalty = float(getattr(transport, "traffic_delay_minutes", 0.0)) * rm
    
    reliability_penalty = float(h.delay_penalty) * rm

    organ_risk_cost = float(organ.risk_weight) * float(risk.risk_score)

    time_pressure = 1.0 - min(1.0, risk.remaining_safe_minutes / max(1, organ.max_transport_minutes))
    urgency_penalty = time_pressure * 12.0

    temperature_penalty = 15.0 if risk.risk_level in ("HIGH", "CRITICAL") else 0.0

    wait_penalty = _expected_wait_penalty(h, risk, organ)
    cap_bonus = _capability_bonus(h, risk)

    total = (
        distance_km
        + reliability_penalty
        + traffic_penalty
        + organ_risk_cost
        + temperature_penalty
        + urgency_penalty
        + wait_penalty
        - cap_bonus
    )

    return {
        "distance_km": distance_km,
        "reliability_penalty": reliability_penalty,
        "organ_risk_cost": organ_risk_cost,
        "temperature_penalty": temperature_penalty,
        "urgency_penalty": urgency_penalty,
        "wait_penalty": wait_penalty,
        "capability_bonus": cap_bonus,  
        "total_cost": total,
        "traffic_penalty": traffic_penalty,
    }

# Evaluate candidate hospitals and select the one with the lowest total cost.
def select_optimal_hospital(
    current: Coord,
    hospitals: List[Hospital],
    risk: RiskState,
    organ: OrganProfile,
    transport: TransportState,
    return_cost: bool = False,
    return_breakdown: bool = False,
) -> Union[
    Hospital,
    Tuple[Hospital, float],
    Tuple[Hospital, float, Dict[str, Dict[str, float]]],
]:
   
    best_h: Optional[Hospital] = None
    best_cost: float = float("inf")
    breakdowns: Dict[str, Dict[str, float]] = {}

    for h in hospitals:
        # If a synthetic road graph is available, compute a risk-aware path cost.
        if hasattr(transport, "road_graph") and transport.road_graph is not None:
            base_path_cost = risk_aware_shortest_path(
                transport.road_graph,
                current,
                h,
                risk,
                organ,
            )

            wait_penalty = _expected_wait_penalty(h, risk, organ)
            cap_bonus = _capability_bonus(h, risk)
            bd = cost_breakdown(current, h, risk, organ, transport)
            bd["base_path_cost"] = float(base_path_cost)
            bd["total_cost"] = float(
                base_path_cost
                + bd["traffic_penalty"]
                + bd["organ_risk_cost"]
                + bd["temperature_penalty"]
                + bd["urgency_penalty"]
                + wait_penalty
                - cap_bonus
            )
            cost = bd["total_cost"]
            breakdowns[h.name] = bd
        else:
            bd = cost_breakdown(current, h, risk, organ, transport)
            cost = bd["total_cost"]
            breakdowns[h.name] = bd

        # Select the hospital with the minimum total routing cost.
        if cost < best_cost:
            best_cost = cost
            best_h = h

    if best_h is None:
        best_h = hospitals[0]
        best_cost = float("inf")

    if return_breakdown:
        return (best_h, best_cost, breakdowns)
    if return_cost:
        return (best_h, best_cost)
    return best_h
