"""Isolation Forest Anomaly Detection Module.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Implements tree-based isolation partitioning across multiple contamination
settings (0.01, 0.03, 0.05) with reproducible seeds and automated artifact export.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_loader import get_project_root
from src.data_preprocessing import NUMERICAL_FEATURE_COLS

logger = logging.getLogger("isolation_forest")
os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".cache" / "matplotlib")
plt.switch_backend("Agg")

CONTAMINATION_LEVELS = [0.01, 0.03, 0.05]
DEFAULT_N_ESTIMATORS = 100
RANDOM_STATE = 42


def train_and_predict_isolation_forest(
    X_scaled: Union[pd.DataFrame, np.ndarray],
    contamination: float = 0.03,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, IsolationForest]:
    """Train Isolation Forest and extract binary labels and anomaly scores.
    
    Parameters
    ----------
    X_scaled : pd.DataFrame or np.ndarray
        Standard-scaled numerical feature matrix.
    contamination : float, default 0.03
        Expected proportion of outliers in data set; controls decision threshold.
    n_estimators : int, default 100
        Number of isolation trees in the ensemble.
    random_state : int, default 42
        Reproducibility seed.
        
    Returns
    -------
    labels : np.ndarray
        Binary labels (1 = anomaly, 0 = normal).
    scores : np.ndarray
        Continuous decision function scores (lower/more negative = more anomalous).
    model : IsolationForest
        Fitted Scikit-learn model instance.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_scaled)
    
    # decision_function returns negative anomaly scores (more negative = anomalous)
    scores = model.decision_function(X_scaled)
    
    # Scikit-learn raw predictions: -1 for outlier, 1 for inlier
    raw_preds = model.predict(X_scaled)
    
    # Map to standardized convention: 1 = anomaly, 0 = normal
    binary_labels = np.where(raw_preds == -1, 1, 0)
    
    return binary_labels, scores, model


def run_isolation_forest_experiments(
    df_processed: Optional[pd.DataFrame] = None,
    contaminations: Optional[List[float]] = None,
    save_outputs: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Execute multi-parameter Isolation Forest experiments.
    
    Iterates across contamination levels (0.01, 0.03, 0.05) and logs detection
    frequencies, threshold shifts, and anomaly intersections.
    """
    root = get_project_root()
    if df_processed is None:
        data_path = root / "data" / "processed" / "ghcnd_processed.csv"
        df_processed = pd.read_csv(data_path, low_memory=False)
        
    if contaminations is None:
        contaminations = CONTAMINATION_LEVELS
        
    scaled_feature_cols = [f"{c}_SCALED" for c in NUMERICAL_FEATURE_COLS]
    X_scaled = df_processed[scaled_feature_cols]
    
    results_df = df_processed.copy()
    experiment_metrics: Dict[str, Dict] = {}
    
    out_dir = root / "outputs" / "anomaly_results"
    graph_dir = root / "graphs" / "isolation_forest"
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    for c in contaminations:
        col_label = f"IF_ANOMALY_C{int(c*100):02d}"
        col_score = f"IF_SCORE_C{int(c*100):02d}"
        
        labels, scores, model = train_and_predict_isolation_forest(
            X_scaled, contamination=c, n_estimators=DEFAULT_N_ESTIMATORS, random_state=RANDOM_STATE
        )
        
        results_df[col_label] = labels
        results_df[col_score] = scores
        
        n_anomalies = int(labels.sum())
        n_normal = len(labels) - n_anomalies
        pct_anomalies = (n_anomalies / len(labels)) * 100.0
        
        experiment_metrics[f"contamination_{c}"] = {
            "contamination": c,
            "n_estimators": DEFAULT_N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "total_observations": len(labels),
            "anomalies_detected": n_anomalies,
            "normal_observations": n_normal,
            "anomaly_percentage": round(pct_anomalies, 2),
            "min_score": float(scores.min()),
            "median_score": float(np.median(scores)),
            "max_score": float(scores.max()),
            "score_threshold_zero": 0.0
        }
        logger.info(
            "IF (c=%.2f): %d anomalies (%.2f%%) detected", c, n_anomalies, pct_anomalies
        )
        
    # Baseline primary label (c = 0.03 as recommended university benchmark)
    results_df["IF_ANOMALY"] = results_df["IF_ANOMALY_C03"]
    results_df["IF_SCORE"] = results_df["IF_SCORE_C03"]
    
    if save_outputs:
        # Save enriched predictions
        out_csv = out_dir / "isolation_forest_results.csv"
        results_df.to_csv(out_csv, index=False)
        logger.info("Saved IF predictions to: %s", out_csv)
        
        # Save summary JSON
        summary_path = root / "outputs" / "evaluation_results" / "isolation_forest_summary.json"
        with open(summary_path, "w") as f:
            json.dump(experiment_metrics, f, indent=2)
            
        # Generate diagnostic plots
        plot_isolation_forest_diagnostics(results_df, graph_dir)
        
    return results_df, experiment_metrics


def plot_isolation_forest_diagnostics(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate diagnostic visualizations for Isolation Forest experiments."""
    sns.set_theme(style="whitegrid", palette="deep")
    
    # 1. Anomaly Score Distribution across Contaminations
    plt.figure(figsize=(9, 5))
    for c in CONTAMINATION_LEVELS:
        col = f"IF_SCORE_C{int(c*100):02d}"
        sns.kdeplot(df[col], label=f"c={c}", fill=True, alpha=0.15)
    plt.axvline(0.0, color="crimson", linestyle="--", linewidth=1.5, label="Decision Boundary (Score=0)")
    plt.title("Isolation Forest Decision Score Distribution", fontsize=13, weight="bold")
    plt.xlabel("Anomaly Score (Negative = Outlier)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "if_score_distributions.png", dpi=200)
    plt.close()
    
    # 2. Temperature Extreme Anomalies Flagged by IF (c=0.03)
    plt.figure(figsize=(10, 6))
    normal = df[df["IF_ANOMALY"] == 0]
    anomalous = df[df["IF_ANOMALY"] == 1]
    
    plt.scatter(normal["TMIN"], normal["TMAX"], c="lightgray", label="Normal", alpha=0.4, s=20)
    plt.scatter(anomalous["TMIN"], anomalous["TMAX"], c="crimson", label="IF Anomaly (c=0.03)", alpha=0.85, s=35, edgecolors="black", linewidths=0.5)
    
    plt.title("Isolation Forest Anomaly Detection on TMIN vs. TMAX Plane", fontsize=13, weight="bold")
    plt.xlabel("Minimum Temperature (°C)", fontsize=11)
    plt.ylabel("Maximum Temperature (°C)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "if_temperature_anomalies.png", dpi=200)
    plt.close()
    
    # 3. Anomaly Count by Station Bar Plot
    plt.figure(figsize=(8, 5))
    station_counts = df[df["IF_ANOMALY"] == 1]["STATION"].value_counts()
    sns.barplot(x=station_counts.index, y=station_counts.values, palette="Reds_r")
    plt.title("Isolation Forest Anomalies by Weather Station (c=0.03)", fontsize=13, weight="bold")
    plt.xlabel("Station Identifier", fontsize=11)
    plt.ylabel("Count of Flagged Anomalies", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "if_anomalies_by_station.png", dpi=200)
    plt.close()
    
    logger.info("Saved IF diagnostic figures to: %s", output_dir)


if __name__ == "__main__":
    print("=" * 70)
    print("Isolation Forest Anomaly Detection (INT396 / SIH26073)")
    print("=" * 70)
    res_df, metrics = run_isolation_forest_experiments()
    print("\nIsolation Forest Experiment Summary:")
    for k, v in metrics.items():
        print(f" - {k}: {v['anomalies_detected']} anomalies ({v['anomaly_percentage']}%)")

