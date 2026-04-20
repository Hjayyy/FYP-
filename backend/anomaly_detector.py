# Lightweight EWMA-based anomaly detector for temperature time-series data.
# Returns a normalised anomaly score in the range [0, 1]

from dataclasses import dataclass


@dataclass
class EWMAAnomalyDetector:

    alpha: float = 0.2
    beta: float = 0.2
    mean: float = 4.0
    var: float = 0.5

    # Updates EWMA mean and variance
    def update(self, x: float) -> float:
        diff = x - self.mean
        self.mean = self.mean + self.alpha * diff
        self.var = (1 - self.beta) * self.var + self.beta * (diff * diff)
       
       # Compute a stable standard deviation estimate
        std = (self.var ** 0.5) if self.var > 1e-9 else 1e-6
        z = abs(x - self.mean) / std
        return min(1.0, z / 6.0) # Normalises score to 0,1
