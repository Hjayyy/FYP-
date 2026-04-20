# Generate simulation logs for each organ type to support
# supervised machine learning dataset construction

from simulation_engine import TransportSimulation

organs = ["heart", "kidney", "lungs"]

for organ in organs:
    print(f"\nGenerating training data for {organ}...")

    sim = TransportSimulation(
        organ_key=organ,
        ml_model_path=None,        #Disable ML for clean data
        delay_probability=0.6      #Increase variability for training
    )

    while not sim.completed:
        sim.step()

    print(f"{organ} simulation complete.")