# Main Simulation Loop
# Integrates temperature sensing, transport dynamics, risk evaluation, and
# adaptive routing into a single real-time simulation pipeline.

from temperature_simulation import TemperatureSimulator
from transport_simulator import TransportSimulator
from risk_engine import RiskEngine
from routing_engine import select_optimal_hospital
from organ_profile import ORGAN_PROFILES
from alert_engine import generate_alert
from simulation_logger import SimulationLogger

# mock hospital locations with gps coordinates

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
    if organ_type not in ORGAN_PROFILES:
        raise ValueError(f"Unsupported organ type: {organ_type}")
    
    print(f"\n--- Starting simulation for {organ_type.upper()} ---\n")

    # initialise components

    temp_sim = TemperatureSimulator()
    transport_sim = TransportSimulator(
        start_location=(51.5074, -0.1278),    # london
        destination=HOSPITALS[0]["location"],
    )

    risk_engine = RiskEngine(organ_type)
    organ_profile = ORGAN_PROFILES[organ_type]
    logger = SimulationLogger()
    
    # simulation loop
    try:
        for minute in range(steps):
            # 1. Sense environment
            temp_data = temp_sim.step()
            transport_data = transport_sim.step()

            # 2. Evaluate risk
            risk_state = risk_engine.evaluate(
                temperature_data=temp_data,
                transport_data=transport_data
            )

            # 3. Generate driver alerts
            alert = generate_alert(risk_state, organ_profile)
            if alert:
                print(f"[DRIVER ALERT] {alert}")

            # 4. Adaptive rerouting if intervention required
            if risk_state["requires_intervention"]:
                optimal_hospital = select_optimal_hospital(
                    current_location=transport_data["location"],
                    hospitals=HOSPITALS,
                    risk_state=risk_state,
                    organ_profile=organ_profile,
                    transport_data=transport_data
                )

                if optimal_hospital:
                    transport_sim.destination = optimal_hospital["location"]
                    print(
                        f"[ROUTING] {risk_state['risk_level']} risk — "
                        f"rerouting to {optimal_hospital['name']}"
                    )

            # 5. Log final state for this minute
            logger.log(
                minute=minute + 1,
                temp=temp_data["temperature"],
                risk=risk_state["risk_level"],
                distance=transport_data["distance_remaining"],
                destination=transport_sim.destination
            )

            # 6. Monitoring output (optional runtime visibility)
            print({
                "minute": minute + 1,
                "temperature": temp_data["temperature"],
                "location": transport_data["location"],
                "risk_level": risk_state["risk_level"],
                "destination": transport_sim.destination
            })

            if transport_data["status"] == "ARRIVED":
                print("\n✓ Transport completed successfully\n")
                break

    finally:
        logger.close()

if __name__ == "__main__":
    run_simulation("lungs", steps=120)
