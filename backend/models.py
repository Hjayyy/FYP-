# Core data structures used across the simulation, including organ profiles,
# hospitals, temperature readings, transport state, and evaluated risk state.

from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, List

#longitude and latitude type
Coord = Tuple[float, float]

# Characteristics & constrains of each organ
@dataclass(frozen=True)
class OrganProfile:
    key: str
    name: str
    safe_temp_range: Tuple[float, float]  # (min, max)
    max_transport_minutes: int
    risk_weight: float
    denature_minutes_margin: int

# Characteristics of a hospital in the network
@dataclass(frozen=True)
class Hospital:
    name: str
    location: Coord
    delay_penalty: float = 0.0
    expected_wait_min: float = 0.0
    capability: float = 0.7

# Temp measurements recorded during transport
@dataclass(frozen=True)
class TemperatureReading:
    timestamp: datetime
    temperature_c: float

# Current transport state of organ
@dataclass(frozen=True)
class TransportState:
    timestamp: datetime
    location: Coord
    elapsed_minutes: int          # total minutes since start (includes delays)
    delay_minutes: int            # how many of those minutes were delays
    distance_remaining_km: float
    status: str                   # "IN_TRANSIT" or "ARRIVED"
    delayed_this_minute: bool
    traffic_state: str = "NORMAL"
    traffic_delay_minutes: int = 0

# Evaluated risk state of organ during transport
@dataclass(frozen=True)
class RiskState:
    timestamp: datetime
    organ_key: str
    risk_score: float
    risk_level: str
    requires_intervention: bool
    anomaly_score: float
    confidence: float
    remaining_safe_minutes: int

    # ML escalations probability 
    escalation_prob: float = 0.0 

    # Displays reasons why risk was triggered for team to understand  
    reasons: List[str] = field(default_factory=list)
