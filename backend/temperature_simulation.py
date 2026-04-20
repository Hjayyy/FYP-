# Simulate organ temperature during transport.
# Models baseline cooling, stochastic variation, breach events,
# mean reversion, and delay-related warming.

import numpy as np
from datetime import datetime, timezone
from models import TemperatureReading, TransportState


class TemperatureSimulator:
    def __init__(
        self,
        base_temp: float = 4.0,

        # Small random sensor noise
        noise_std: float = 0.08,

        # small random walk drift (per minute)
        drift_std: float = 0.001,

        # Warm breach events (most common)
        warm_breach_probability: float = 0.08,
        warm_breach_min: float = 0.6,
        warm_breach_max: float = 2.0,
        warm_breach_duration_min: int = 2,
        warm_breach_duration_max: int = 6,

        # Cold breach events (rare)
        cold_breach_probability: float = 0.002,
        cold_breach_min: float = 0.3,
        cold_breach_max: float = 1.0,
        cold_breach_duration_min: int = 2,
        cold_breach_duration_max: int = 5,

        # Physical temp bounds
        min_temp: float = 2.0,
        max_temp: float = 8.0,

        # Pull temperature back towards the cooling baseline.
        mean_reversion: float = 0.05,

        # Extra warming during delays
        delay_warming_per_min: float = 0.06,
    ):
        self.base_temp = float(base_temp)
        self.noise_std = float(noise_std)
        self.drift_std = float(drift_std)

        # Warm breach parameters
        self.warm_breach_probability = float(warm_breach_probability)
        self.warm_breach_min = float(warm_breach_min)
        self.warm_breach_max = float(warm_breach_max)
        self.warm_breach_duration_min = int(warm_breach_duration_min)
        self.warm_breach_duration_max = int(warm_breach_duration_max)

        # Cold breach parameters
        self.cold_breach_probability = float(cold_breach_probability)
        self.cold_breach_min = float(cold_breach_min)
        self.cold_breach_max = float(cold_breach_max)
        self.cold_breach_duration_min = int(cold_breach_duration_min)
        self.cold_breach_duration_max = int(cold_breach_duration_max)

        # Temp bounds
        self.min_temp = float(min_temp)
        self.max_temp = float(max_temp)
        
        # Stabilisation toward base temp
        self.mean_reversion = float(mean_reversion)

        # Warms organ during transport delays
        self.delay_warming_per_min = float(delay_warming_per_min)

        self.current_temp = float(base_temp)

        # Internal state for ongoing breach events
        self._warm_remaining = 0
        self._cold_remaining = 0
        self._warm_strength = 0.0
        self._cold_strength = 0.0

    def apply_intervention(self, cooling_delta: float = 0.8) -> None:
        
        # Apply a cooling intervention to reduce the current temperature.
        self.current_temp -= cooling_delta
        # Clamp to realistic physiological lower bound
        if self.current_temp < 2.0:
            self.current_temp = 2.0

    def _maybe_start_events(self) -> None:
        # Trigger new warm or cold breach events probabilistically
        if self._warm_remaining <= 0 and np.random.rand() < self.warm_breach_probability:
            self._warm_remaining = np.random.randint(self.warm_breach_duration_min, self.warm_breach_duration_max + 1)
            self._warm_strength = float(np.random.uniform(self.warm_breach_min, self.warm_breach_max))

        if self._cold_remaining <= 0 and np.random.rand() < self.cold_breach_probability:
            self._cold_remaining = np.random.randint(self.cold_breach_duration_min, self.cold_breach_duration_max + 1)
            self._cold_strength = float(np.random.uniform(self.cold_breach_min, self.cold_breach_max))

    # Advance the temperature simulation by one timestep.
    def step(self, transport: TransportState | None = None) -> TemperatureReading:
        self._maybe_start_events() 

        # Baseline temperature dynamics
        noise = float(np.random.normal(0.0, self.noise_std))
        drift = float(np.random.normal(0.0, self.drift_std))

        # Extra warming during delay
        delay_warming = 0.0
        if transport is not None and getattr(transport, "delayed_this_minute", False):
            
           # Warming increases slightly if temp already elevated
            factor = 1.0 + max(0.0, (self.current_temp - self.base_temp) / 4.0)
            delay_warming = self.delay_warming_per_min * factor

        # Warm breach contribution
        warm = 0.0
        if self._warm_remaining > 0:
            warm = self._warm_strength
            self._warm_remaining -= 1
            self._warm_strength *= 0.65 

        # Cold breach contribution 
        cold = 0.0
        if self._cold_remaining > 0:
            cold = -self._cold_strength
            self._cold_remaining -= 1
            self._cold_strength *= 0.65

        # Mean reversion towards the cooling baseline
        pull = (self.base_temp - self.current_temp) * self.mean_reversion

        # Update temperature
        self.current_temp += noise + drift + delay_warming + warm + cold + pull

        # Clamp temperature within physiological bounds
        self.current_temp = min(self.max_temp, max(self.min_temp, self.current_temp))

        return TemperatureReading(
            timestamp=datetime.now(timezone.utc),
            temperature_c=round(self.current_temp, 2),
        )
