# Main Simulation Loop
# Integrates temperature sensing, transport dynamics, risk evaluation, and
# adaptive routing into a single real-time simulation pipeline.

from temperature_simulation import TemperatureSimulator
from transport_simulator import TransportSimulator
from risk_engine import RiskEngine
from routing_engine import select_optimal_hospital
from organ_profile import ORGAN_PROFILES

# mock hospital locations

HOSPITALS = [
    {
        "name": "Hospital A",
        "location": (51.4700, -0.4543)  # Heathrow area
    },
    {
        "name": "Hospital B",
        "location": (51.4545, -2.5879)  # Bristol
    },
    {
        "name": "Hospital C",
        "location": (52.4862, -1.8904)  # Birmingham
    }
]

def run_simulation(organ_type: str, steps: int = 60):
    print(f"\n--- Starting simulation for {organ_type.upper()} ---\n")

    # initialise components

    temp_sim = TemperatureSimulator()
    transport_sim = TransportSimulator(
        start_location=(51.5074, -0.1278),
        destination=HOSPITALS[0]["location"],
    )

    risk_engine = RiskEngine(organ_type)
    organ_profile = ORGAN_PROFILES[organ_type]

    # simulation loop

    for minute in range(steps):
        temp_data = temp_sim.step()
        transport_data = transport_sim.step()

        risk_state = risk_engine.evaluate(
            temperature_data=temp_data,
            transport_data=transport_data,
        )

        # adapting rerouting logic

        if risk_state["requires_intervention"]:
            optimal_hospital = select_optimal_hospital(
                current_location=transport_data["location"],
                hospitals=HOSPITALS,
                risk_state=risk_state,
                organ_profile=organ_profile,
            )

            if optimal_hospital:
                transport_sim.destination = optimal_hospital["location"]

                print(
                    f"[ALERT] {risk_state['risk_level']} risk detected — "
                    f"rerouting to {optimal_hospital['name']}"
                )

        # monitoring output

        print({
            "minute": minute + 1,
            "temperature": temp_data["temperature"],
            "transport_data": transport_data["location"],
            "risk_level": risk_state["risk_level"],
            "destination": transport_sim.destination
        })

        if transport_data["status"] == "ARRIVED":
            print("\n✓ Transport completed\n")
            break

    if __name__ == "__main__":
        run_simulation("lungs", steps=120)
