# cold chain condition simulation, models/ reflects its behaviour
# with accounts for natural fluctuations, gradual drifts and occasional
# breach events

import numpy as np
from datetime import datetime, timezone


class TemperatureSimulator:
    def __init__(
        self,
        base_temp=4.0,
        noise_std=0.2,
        drift_rate=0.01,
        breach_probability=0.02
    ):
        # the baseline temp for cold organ storage
        self.base_temp = base_temp
        self.noise_std = noise_std # standard deviation for normal temp noise
        self.drift_rate = drift_rate #temp drift per step
        self.breach_probability = breach_probability # chance of breach happening
        self.current_temp = base_temp

    def step(self):
        # normal temp fluctuation
        noise = np.random.normal(0, self.noise_std) # random noise to represent sensor variation
        drift = np.random.choice([-1, 1]) * self.drift_rate # drift to model warming or cooling (gradual)

        # for the occasional temperature breach
        if np.random.rand() < self.breach_probability:
            breach = np.random.uniform(2.0, 6.0)
        else:
            breach = 0.0
        # updates to current temp
        self.current_temp += noise + drift + breach

        # outputs structured sensor reading
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "temperature": float(round(self.current_temp, 2))
        }
# Testing code
# if __name__ == "__main__":
    # sim = TemperatureSimulator()
    # for _ in range(5):
        # print(sim.step())
