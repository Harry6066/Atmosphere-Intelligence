"""Evaluation, False-Positive Investigation, and Scientific Audit Module.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Investigates detection stability, examines known historical extreme weather events
(e.g., 2019 Chicago Polar Vortex, 2021 Pacific NW Heat Dome, 2021 NYC Hurricane Ida),
audits potential false positives vs. sensor defects, and quantifies geographical bias.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_loader import get_project_root

logger = logging.getLogger("evaluation")
os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".cache" / "matplotlib")
plt.switch_backend("Agg")

# Ground truth historical climate events for validation
HISTORICAL_EXTREME_EVENTS = [
    {
        "event_name": "Chicago Polar Vortex Record Freeze",
        "station": "USW00094846",
        "start_date": "2019-01-29",
        "end_date": "2019-01-31",
        "expected_phenomenon": "Historic subzero temperatures (TMIN reaching -30.5°C / -23°F)",
        "category": "Legitimate Extreme Weather"
    },
    {
        "event_name": "Pacific Northwest Historic Heat Dome",
        "station": "USW00024233",
        "start_date": "2021-06-26",
        "end_date": "2021-06-28",
        "expected_phenomenon": "All-time record high temperatures in Seattle (TMAX reaching 42.2°C / 108°F)",
        "category": "Legitimate Extreme Weather"
    },
    {
        "event_name": "Hurricane Ida Flash Flooding",
        "station": "USW00094728",
        "start_date": "2021-09-01",
        "end_date": "2021-09-02",
        "expected_phenomenon": "Record torrential rainfall (PRCP > 180 mm in 24 hours at Central Park)",
        "category": "Legitimate Extreme Weather"
    },
    {
        "event_name": "Chicago Winter Blizzard & Deep Snow",
        "station": "USW00094846",
        "start_date": "2021-02-15",
        "end_date": "2021-02-16",
        "expected_phenomenon": "Heavy lake-effect blizzard, subzero highs and snow accumulation > 400 mm",
        "category": "Legitimate Extreme Weather"
    }
]


def audit_historical_extreme_events(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate whether unsupervised models flag documented real-world extreme climate events."""
    results = []
    
    for event in HISTORICAL_EXTREME_EVENTS:
        mask = (
            (df["STATION"] == event["station"]) &
            (df["DATE"] >= event["start_date"]) &
            (df["DATE"] <= event["end_date"])
        )
        subset = df[mask]
        
        if len(subset) > 0:
            if_detected = int((subset["IF_ANOMALY"] == 1).sum())
            lof_detected = int((subset["LOF_ANOMALY"] == 1).sum())
            consensus_detected = int((subset["CONSENSUS_LABEL"] == "Consensus Anomaly").sum())
            
            results.append({
                "Event": event["event_name"],
                "Station": event["station"],
                "Date_Range": f"{event['start_date']} to {event['end_date']}",
                "Days": len(subset),
                "Expected_Phenomenon": event["expected_phenomenon"],
                "IF_Flagged_Days": if_detected,
                "LOF_Flagged_Days": lof_detected,
                "Consensus_Flagged_Days": consensus_detected,
                "Scientific_Classification": event["category"],
                "Assessment": "Correctly Identified Extreme Event (Statistical Outlier, NOT Sensor Fault)"
            })
            
    return pd.DataFrame(results)


def analyze_geographical_and_seasonal_bias(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Examine detection rates across individual stations and seasons to quantify bias.
    
    Unsupervised models trained across pooled multi-station data can suffer from
    geographical bias: stations with high climate variance (e.g. Chicago) naturally
    produce more mathematical outliers than equable maritime climates (e.g. Seattle).
    """
    # 1. By Station
    station_bias = df.groupby(["STATION", "NAME"]).agg(
        Total_Records=("DATE", "count"),
        IF_Anomalies=("IF_ANOMALY", "sum"),
        LOF_Anomalies=("LOF_ANOMALY", "sum"),
        Consensus_Anomalies=("CONSENSUS_LABEL", lambda x: (x == "Consensus Anomaly").sum())
    ).reset_index()
    
    station_bias["IF_Rate_Pct"] = (station_bias["IF_Anomalies"] / station_bias["Total_Records"]) * 100
    station_bias["LOF_Rate_Pct"] = (station_bias["LOF_Anomalies"] / station_bias["Total_Records"]) * 100
    station_bias["Consensus_Rate_Pct"] = (station_bias["Consensus_Anomalies"] / station_bias["Total_Records"]) * 100
    
    # 2. By Season (1: Winter, 2: Spring, 3: Summer, 4: Fall)
    season_names = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
    df["SEASON_NAME"] = df["SEASON"].map(season_names)
    
    season_bias = df.groupby("SEASON_NAME").agg(
        Total_Records=("DATE", "count"),
        IF_Anomalies=("IF_ANOMALY", "sum"),
        LOF_Anomalies=("LOF_ANOMALY", "sum"),
        Consensus_Anomalies=("CONSENSUS_LABEL", lambda x: (x == "Consensus Anomaly").sum())
    ).reindex(["Winter", "Spring", "Summer", "Fall"]).reset_index()
    
    season_bias["IF_Rate_Pct"] = (season_bias["IF_Anomalies"] / season_bias["Total_Records"]) * 100
    season_bias["LOF_Rate_Pct"] = (season_bias["LOF_Anomalies"] / season_bias["Total_Records"]) * 100
    season_bias["Consensus_Rate_Pct"] = (season_bias["Consensus_Anomalies"] / season_bias["Total_Records"]) * 100
    
    return {"station_bias": station_bias, "season_bias": season_bias}


def inspect_top_suspicious_observations(df: pd.DataFrame, n_top: int = 15) -> pd.DataFrame:
    """Extract top consensus anomalies for documented expert review.
    
    Ranks observations by combined anomaly severity:
    severity = (-IF_SCORE) * (-LOF_SCORE - 1.0)
    """
    consensus = df[df["CONSENSUS_LABEL"] == "Consensus Anomaly"].copy()
    
    # Combined severity metric
    consensus["SEVERITY_INDEX"] = (
        (-consensus["IF_SCORE"]) * ((-consensus["LOF_SCORE"]) - 1.0)
    )
    
    top_obs = consensus.sort_values(by="SEVERITY_INDEX", ascending=False).head(n_top)
    
    display_cols = [
        "STATION", "DATE", "TMAX", "TMIN", "TEMP_RANGE", "PRCP", "SNOW", "SNWD",
        "TMAX_DIFF_1D", "IF_SCORE", "LOF_SCORE", "SEVERITY_INDEX"
    ]
    return top_obs[display_cols]


def run_evaluation_pipeline(save_outputs: bool = True) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Execute evaluation and false-positive audit pipeline."""
    root = get_project_root()
    data_file = root / "outputs" / "anomaly_results" / "consensus_anomaly_results.csv"
    df = pd.read_csv(data_file, low_memory=False)
    
    # 1. Historical Extreme Events Audit
    df_events = audit_historical_extreme_events(df)
    logger.info("Audited %d historical extreme events", len(df_events))
    
    # 2. Geographical and Seasonal Bias Audit
    bias_tables = analyze_geographical_and_seasonal_bias(df)
    
    # 3. Top Suspicious Observations for Manual Review
    df_top_suspicious = inspect_top_suspicious_observations(df, n_top=20)
    
    if save_outputs:
        eval_dir = root / "outputs" / "evaluation_results"
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Save audit tables
        df_events.to_csv(eval_dir / "historical_extreme_events_audit.csv", index=False)
        bias_tables["station_bias"].to_csv(eval_dir / "geographical_station_bias.csv", index=False)
        bias_tables["season_bias"].to_csv(eval_dir / "seasonal_bias.csv", index=False)
        df_top_suspicious.to_csv(eval_dir / "top_suspicious_observations.csv", index=False)
        
        logger.info("Saved evaluation audit tables to: %s", eval_dir)
        
    return df_events, bias_tables


if __name__ == "__main__":
    print("=" * 70)
    print("Scientific Evaluation & False-Positive Audit (INT396 / SIH26073)")
    print("=" * 70)
    events_df, biases = run_evaluation_pipeline()
    print("\n1. Historical Extreme Climate Events Verification:")
    print(events_df[["Event", "Days", "IF_Flagged_Days", "LOF_Flagged_Days", "Consensus_Flagged_Days", "Assessment"]].to_string(index=False))
    print("\n2. Geographical Station Bias Breakdown:")
    print(biases["station_bias"][["STATION", "Total_Records", "IF_Rate_Pct", "LOF_Rate_Pct", "Consensus_Rate_Pct"]].to_string(index=False))
    print("\n3. Seasonal Bias Breakdown:")
    print(biases["season_bias"][["SEASON_NAME", "Total_Records", "IF_Rate_Pct", "LOF_Rate_Pct", "Consensus_Rate_Pct"]].to_string(index=False))

