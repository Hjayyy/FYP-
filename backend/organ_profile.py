# Define organ-specific transport and preservation constraints
# used throughout the simulation and risk evaluation process

from models import OrganProfile

# Dictionary of organ profiles used by the sim system which influence risk and
# routing decisoons
ORGAN_PROFILES = {

    # Heart transplant constraints
    "heart": OrganProfile(
        key="heart",
        name="Heart",
        safe_temp_range=(4.0, 8.0),  # The clinically accepted range
        max_transport_minutes=240,
        risk_weight=1.5, # High due to organ sensitivity
        denature_minutes_margin=30,
    ),

    # Lung transplant constrains
    "lungs": OrganProfile(
        key="lungs",
        name="Lungs",
        safe_temp_range=(4.0, 8.0), # Standard cold storage range 
        max_transport_minutes=360,
        risk_weight=1.3,
        denature_minutes_margin=30,
    ),

    # Kidney transplant constraints
    "kidney": OrganProfile(
        key="kidney",
        name="Kidney",
        safe_temp_range=(2.0, 8.0), # Typical hypothermic storage 
        max_transport_minutes=720,
        risk_weight=0.8, 
        denature_minutes_margin=45,
    ),
}
