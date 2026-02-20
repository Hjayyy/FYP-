from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, List

Coord = Tuple[float, float]


@dataclass(frozen=True)
class OrganProfile:
    key: str
    name: str
    safe_temp_range: Tuple[float, float]  # (min, max)
    max_transport_minutes: int
    risk_weight: float
    denature_minutes_margin: int


@dataclass(frozen=True)
class Hospital:
    name: str
    location: Coord
    delay_penalty: float = 0.0


@dataclass(frozen=True)
class TemperatureReading:
    timestamp: datetime
    temperature_c: float


@dataclass(frozen=True)
class TransportState:
    timestamp: datetime
    location: Coord
    elapsed_minutes: int          # total minutes since start (includes delays)
    delay_minutes: int            # how many of those minutes were delays
    distance_remaining_km: float
    status: str                   # "IN_TRANSIT" | "ARRIVED"
    delayed_this_minute: bool


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
    reasons: List[str] = field(default_factory=list)
