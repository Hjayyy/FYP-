# Simulate vehicle movement between hospitals, including travel progress,
# arrival detection, and stochastic traffic delays

import random
import math
from datetime import datetime, timezone
from models import TransportState, Coord

# Compute the geographical distance between two coordinates
# using the Haversine formula
def haversine(coord1: Coord, coord2: Coord) -> float:
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    r = 6371.0  # radius of the earth

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    # Haversine distance calculation
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Simulate the current traffic state and associated delay severity.
def simulate_traffic_state() -> tuple[str, int, float]:

    states = [
        ("NORMAL", 0, 1.0),
        ("CONGESTED", random.randint(1, 3), 1.3),
        ("SEVERE", random.randint(2, 5), 1.8),
        ("INCIDENT", random.randint(4, 8), 2.5),
    ]
    return random.choices(
        states,
        weights=[0.55, 0.25, 0.15, 0.05],
        k=1
    )[0]


class TransportSimulator:
    def __init__(
        self,
        start_location: Coord,
        destination: Coord,
        speed_km_per_minute: float = 1.0,
        delay_probability: float = 0.02,
        min_delay_window: int = 1,
        max_delay_window: int = 3,
        arrive_threshold_km: float = 0.25
    ):
        # Current vehicle location.
        self.current_location = start_location

        # Destination hospital
        self.destination = destination
        self.speed = float(speed_km_per_minute)  # km per simulation minute
        self.delay_probability = float(delay_probability)  # chance of getting in traffic
        self.min_delay_window = int(min_delay_window)
        self.max_delay_window = int(max_delay_window)
        self.arrive_threshold_km = float(arrive_threshold_km)

        # Transport state tracking
        self.elapsed_minutes = 0
        self.delay_minutes = 0
        self.status = "IN_TRANSIT"
        self._delay_countdown = 0

        # Real time traffic management
        self.traffic_state = "NORMAL"
        self.traffic_delay_minutes = 0
        self.traffic_reliability_factor = 1.0

    def set_destination(self, destination: Coord) -> None:
        self.destination = destination
        if self.status == "ARRIVED":
            self.status = "IN_TRANSIT"
        self._delay_countdown = 0

    # Update the transport destination, used during rerouting
    def step(self) -> TransportState:
        self.elapsed_minutes += 1
        remaining = haversine(self.current_location, self.destination)

        # Arrival check
        if remaining <= self.arrive_threshold_km:
            self.current_location = self.destination
            self.status = "ARRIVED"
            
            return TransportState(
                timestamp=datetime.now(timezone.utc),
                location=self.current_location,
                elapsed_minutes=self.elapsed_minutes,
                delay_minutes=self.delay_minutes,
                distance_remaining_km=0.0,
                status=self.status,
                delayed_this_minute=False,
                traffic_state="NORMAL",
                traffic_delay_minutes=0
            )

        # Keep the vehicle stationary during an active delay window
        if self._delay_countdown > 0:
            self._delay_countdown -= 1
            self.delay_minutes += 1
            remaining = haversine(self.current_location, self.destination)

            return TransportState(
                timestamp=datetime.now(timezone.utc),
                location=self.current_location,
                elapsed_minutes=self.elapsed_minutes,
                delay_minutes=self.delay_minutes,
                distance_remaining_km=round(remaining, 2),
                status=self.status,
                delayed_this_minute=True,
                traffic_state="DELAY_ACTIVE",
                traffic_delay_minutes=self._delay_countdown + 1
            )

        # Only generate new traffic state when not already delayed
        self.traffic_state, extra_delay, self.traffic_reliability_factor = simulate_traffic_state()
        self.traffic_delay_minutes = extra_delay

        adjusted_delay_probability = self.delay_probability * self.traffic_reliability_factor

        # Trigger a new delay event probabilistically
        if remaining > (self.arrive_threshold_km * 4) and random.random() < adjusted_delay_probability:
            window = random.randint(
                self.min_delay_window + self.traffic_delay_minutes,
                self.max_delay_window + self.traffic_delay_minutes
            )
            self._delay_countdown = max(0, window - 1)
            self.delay_minutes += 1

            return TransportState(
                timestamp=datetime.now(timezone.utc),
                location=self.current_location,
                elapsed_minutes=self.elapsed_minutes,
                delay_minutes=self.delay_minutes,
                distance_remaining_km=round(remaining, 2),
                status=self.status,
                delayed_this_minute=True,
                traffic_state=self.traffic_state,
                traffic_delay_minutes=window
            )

        # Move the vehicle towards the destination
        lat, lon = self.current_location
        dest_lat, dest_lon = self.destination

        step_fraction = min(1.0, self.speed / max(remaining, 1e-6))
        lat += (dest_lat - lat) * step_fraction
        lon += (dest_lon - lon) * step_fraction
        self.current_location = (lat, lon)

        remaining = haversine(self.current_location, self.destination)

        if remaining <= self.arrive_threshold_km:
            self.current_location = self.destination
            self.status = "ARRIVED"
            remaining = 0.0

        return TransportState(
            timestamp=datetime.now(timezone.utc),
            location=self.current_location,
            elapsed_minutes=self.elapsed_minutes,
            delay_minutes=self.delay_minutes,
            distance_remaining_km=round(remaining, 2),
            status=self.status,
            delayed_this_minute=False,
            traffic_state=self.traffic_state,
            traffic_delay_minutes=0
        )
