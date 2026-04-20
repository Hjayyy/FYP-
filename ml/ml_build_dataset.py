# Define organ-specific transport and preservation constraints
# used throughout the simulation and risk evaluation process

from __future__ import annotations

import csv
from typing import List, Dict

RISK_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}


def _will_escalate(levels: List[str], idx: int, horizon: int) -> int:
    
    # Define organ-specific transport and preservation constraints
    # used throughout the simulation and risk evaluation process
    end = min(len(levels), idx + horizon + 1)
    for j in range(idx + 1, end):
        if levels[j] in ("HIGH", "CRITICAL"):
            return 1
    return 0

# Detect the CSV delimiter automatically
def _sniff_dialect(path: str) -> csv.Dialect:
    with open(path, "r", newline="") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
    except csv.Error:
        return csv.get_dialect("excel")

# Clean CSV header names
def _normalize_row_keys(row: Dict[str, str]) -> Dict[str, str]:

    clean = {(k or "").strip().replace("\ufeff", ""): v for k, v in row.items()}
    clean.pop("", None)
    return clean

# Converts simulation_log.csv into labelled ML dataset
# supports older log formats through colummn aliases
def build_dataset(sim_log_csv: str, out_csv: str, horizon_minutes: int = 10) -> None:

    dialect = _sniff_dialect(sim_log_csv)

    rows: List[Dict[str, str]] = []
    with open(sim_log_csv, "r", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None:
            raise RuntimeError("Could not read header row from log CSV.")

        for r in reader:
            rows.append(_normalize_row_keys(r))

    if not rows:
        raise RuntimeError("Empty simulation log.")

    # Show columns if something goes wrong
    available_cols = set(rows[0].keys())

    # Map older column names to new ones
    alias = {
        "temp_c": "temperature_c",
        "temperature": "temperature_c",
        "anomaly": "anomaly_score",
    }

    def get_val(r: Dict[str, str], key: str) -> str:
        # Resolve aliases for older column names
        for old, new in alias.items():
            if key == new and old in r and new not in r:
                return r[old]

        # If elapsed_minutes missing, fall back to minute
        if key == "elapsed_minutes" and "elapsed_minutes" not in r and "minute" in r:
            return r["minute"]

        if key not in r:
            raise KeyError(
                f"Missing column '{key}' in log. Available columns: {sorted(available_cols)}"
            )
        return r[key]

    levels = [get_val(r, "risk_level") for r in rows]

    feature_cols = [
        "temperature_c",
        "elapsed_minutes",          
        "delay_minutes",
        "delayed_this_minute",
        "distance_remaining_km",
        "risk_score",
        "anomaly_score",            
        "confidence",
        "remaining_safe_minutes",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(feature_cols + ["label_escalate", "risk_level_now"])

        for i, r in enumerate(rows):
            x = [get_val(r, c) for c in feature_cols]

            current_level = get_val(r, "risk_level")

    # Only label escalation if current state is NOT already high/critical
            if current_level in ("HIGH", "CRITICAL"):
                y = 0
            else:
                y = _will_escalate(levels, i, horizon_minutes)

            writer.writerow(x + [y, current_level])
    
    # Save the dataset and report the detected structure
    print(f"Saved dataset to {out_csv}")
    print(f"Detected delimiter: {repr(dialect.delimiter)}")
    print(f"Columns found: {sorted(available_cols)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input simulation_log.csv")
    ap.add_argument("--out", dest="out", required=True, help="Output dataset csv")
    ap.add_argument("--horizon", type=int, default=10, help="Lookahead minutes for label")
    args = ap.parse_args()

    build_dataset(args.inp, args.out, horizon_minutes=args.horizon)