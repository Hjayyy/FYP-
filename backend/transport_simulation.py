# transport and location dynamics, where the organ is, how it moves and whether it deviates
# route progression, estimated tie remaining, transportation status
# delayed events (traffic, weather customs) and risk calc if delayed
# Transport progression is simulated using discrete time steps with probabilistic delay events,
# producing structured outputs suitable for real-time monitoring and downstream risk analysis.
import random
import math
from datetime import datetime, timezone


# this calculates the dist in km between 2 GPS points
def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    r = 6371 # earths radium in km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )

    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# models the physical movement of an organ from origin to destination
class TransportSimulator:
    def __init__(
        self,
        start_location,
        destination,
        speed_km_per_minute=1.0,
        delay_probability=0.05,
    ):

        # current gps location (x, y)
        self.current_location = start_location
        # final planned destination
        self.destination = destination
        # transport speed per minute
        self.speed = speed_km_per_minute
        # delay modelling
        self.delay_probability = delay_probability


        #time tracking
        self.elapsed_minutes = 0
        self.delay_minutes = 0
        self.status = "IN_TRANSIT"

    # advance traffic by one minute
    def step(self):

        self.elapsed_minutes += 1

        # considers random delay like traffic & weather
        if random.random() < self.delay_probability:
            self.delay_minutes += random.randint(5, 20)

        # move towards destination, simplified linear movement
        lat, lon = self.current_location
        dest_lat, dest_lon = self.destination

        lat += (dest_lat - lat) * 0.01
        lon += (dest_lon - lon) * 0.01

        self.current_location = (lat, lon)

        remaining_distance = haversine(
            self.current_location, self.destination
        )

        if remaining_distance < 1:
            self.status = "ARRIVED"


        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "location": self.current_location,
            "elapsed_minutes": self.elapsed_minutes,
            "delay_minutes": self.delay_minutes,
            "distance_remaining": round(remaining_distance, 2),
            "status": self.status

        }
# testing code
if __name__ == "__main__":
    sim = TransportSimulator(
        start_location=(51.5074, -0.1278),    # London
        destination=(51.4545, -2.5879)          # Bristol
    )
    for _ in range(10):
         print(sim.step())
