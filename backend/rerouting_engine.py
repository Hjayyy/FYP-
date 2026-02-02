# dynamically selects the optimal hospital destination based on
# transport risk and organ sensitivity
# Rather than relying on static shortest-path algorithms, this routing engine
# dynamically evaluates hospital destinations using distance, organ sensitivity,
# and real-time risk levels. This enables proactive rerouting to preserve organ viability

from transport_simulator import haversine

def compute_route_cost(current_location, hospital, risk_state, organ_profile):

    distance_cost = haversine(
        current_location, hospital["location"]
    )

    # organ sensitivity weighting
    organ_risk_cost = organ_profile["risk_weight"] * risk_state["risk_score"]

    # temp risk escalation
    temperature_penalty = 0
    if risk_state["risk_level"] in ["HIGH, CRITICAL"]:
        temperature_penalty = 15

    total_cost = distance_cost + organ_risk_cost + temperature_penalty

    return total_cost

# selecting the hospital with the lowest dynamic routing cost
def select_optimal_hospital(
    current_location,
    hospitals,
    risk_state,
    organ_profile
    ):

    best_hospital = None
    lowest_cost = float("inf")

    for hospital in hospitals:
        cost = compute_route_cost(
            current_location,
            hospital,
            risk_state,
            organ_profile
        )

        if cost < lowest_cost:
            lowest_cost = cost
            best_hospital = hospital

    return best_hospital
