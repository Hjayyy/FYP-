# Log simulation timestep data to CSV for evaluation and machine learning use

from __future__ import annotations

import csv
import os
from typing import Optional

from models import TemperatureReading, TransportState, RiskState, Coord


class SimulationLogger:

    # Initialise the logger and create the CSV header if the file is new
    def __init__(self, filename: str = "simulation_log.csv"):

        # Check if the file already exists
        new_file = not os.path.exists(filename)

        # Open the CSV file in append mode
        self.file = open(filename, "a", newline="")

        # CSV writer used to add rows
        self.writer = csv.writer(self.file)

        # If this is a new file, write the column headers
        if new_file:
            self.writer.writerow([
                "minute",
                "temperature_c",
                "elapsed_minutes",
                "delay_minutes",
                "delayed_this_minute",
                "lat",
                "lon",
                "distance_remaining_km",
                "risk_score",
                "risk_level",
                "confidence",
                "anomaly_score",
                "remaining_safe_minutes",
                "escalation_prob",
                "destination_lat",
                "destination_lon",
                "routing_cost",
            ])


    # Records one timestep of simulation data
    def log(
        self,
        minute: int,
        temp: TemperatureReading,
        risk: RiskState,
        transport: TransportState,
        destination: Coord,
        routing_cost: Optional[float] = None,
    ) -> None:

        dest_lat, dest_lon = destination
        lat, lon = transport.location

        # Write the current simulation state as a CSV row
        self.writer.writerow([
            minute,
            temp.temperature_c,
            transport.elapsed_minutes,
            transport.delay_minutes,
            int(bool(transport.delayed_this_minute)),
            lat,
            lon,
            transport.distance_remaining_km,
            risk.risk_score,
            risk.risk_level,
            risk.confidence,
            risk.anomaly_score,
            risk.remaining_safe_minutes,
            risk.escalation_prob,
            dest_lat,
            dest_lon,
            "" if routing_cost is None else round(float(routing_cost), 6),
        ])


    # Close the log file when simulation completes.
    def close(self) -> None:
        self.file.close()