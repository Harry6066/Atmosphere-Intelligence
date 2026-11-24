"""Local Outlier Factor (LOF) Anomaly Detection Module.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Implements local density-based reachability outlier detection across multiple
neighborhood parameter configurations (k = 10, 20, 30) using Scikit-learn.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_loader import get_project_root
from src.data_preprocessing import NUMERICAL_FEATURE_COLS

logger = logging.getLogger("lof_detection")
os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".cache" / "matplotlib")
plt.switch_backend("Agg")

NEIGHBORHOOD_SIZES = [10, 20, 30]
DEFAULT_CONTAMINATION = 0.03


def train_and_predict_lof(
    X_scaled: Union[pd.DataFrame, np.ndarray],
    n_neighbors: int = 20,
    contamination: float = DEFAULT_CONTAMINATION
) -> Tuple[np.ndarray, np.ndarray, LocalOutlierFactor]:
    """Execute Local Outlier Factor batch detection.
    
    Parameters
    ----------
    X_scaled : pd.DataFrame or np.ndarray
        Standard-scaled numerical feature matrix.
    n_neighbors : int, default 20
        Number of neighbors for k-distance and local reachability density calculation.
    contamination : float, default 0.03
        Expected proportion of outliers; determines the decision threshold.
        
    Returns
    -------
    labels : np.ndarray
        Binary labels (1 = anomaly, 0 = normal).
    negative_scores : np.ndarray
        Negative outlier factors (closer to -1.0 is normal, substantially lower is outlier).
    model : LocalOutlierFactor
        Fitted Scikit-learn LOF model.
    """
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
        n_jobs=-1
    )
    
    # In batch mode (novelty=False), fit_predict returns inlier (+1) or outlier (-1)
    raw_preds = model.fit_predict(X_scaled)
    negative_scores = model.negative_outlier_factor_
    
    # Standardize convention: 1 = anomaly, 0 = normal
    binary_labels = np.where(raw_preds == -1, 1, 0)
    
    return binary_labels, negative_scores, model


def run_lof_experiments(
    df_processed: Optional[pd.DataFrame] = None,
    neighbor_configs: Optional[List[int]] = None,
    contamination: float = DEFAULT_CONTAMINATION,
    save_outputs: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Execute multi-parameter LOF experiments across neighborhood sizes k in [10, 20, 30]."""
    root = get_project_root()
    if df_processed is None:
        data_path = root / "data" / "processed" / "ghcnd_processed.csv"
        df_processed = pd.read_csv(data_path, low_memory=False)
        
    if neighbor_configs is None:
        neighbor_configs = NEIGHBORHOOD_SIZES
        
    scaled_feature_cols = [f"{c}_SCALED" for c in NUMERICAL_FEATURE_COLS]
    X_scaled = df_processed[scaled_feature_cols]
    
    results_df = df_processed.copy()
    experiment_metrics: Dict[str, Dict] = {}
    
    out_dir = root / "outputs" / "anomaly_results"
    graph_dir = root / "graphs" / "lof"
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    for k in neighbor_configs:
        col_label = f"LOF_ANOMALY_K{k}"
        col_score = f"LOF_SCORE_K{k}"
        
        labels, scores, model = train_and_predict_lof(
            X_scaled, n_neighbors=k, contamination=contamination
        )
        
        results_df[col_label] = labels
        results_df[col_score] = scores
        
        n_anomalies = int(labels.sum())
        n_normal = len(labels) - n_anomalies
        pct_anomalies = (n_anomalies / len(labels)) * 100.0
        
        experiment_metrics[f"neighbors_{k}"] = {
            "n_neighbors": k,
            "contamination": contamination,
            "novelty": False,
            "total_observations": len(labels),
            "anomalies_detected": n_anomalies,
            "normal_observations": n_normal,
            "anomaly_percentage": round(pct_anomalies, 2),
            "min_score": float(scores.min()),
            "median_score": float(np.median(scores)),
            "max_score": float(scores.max()),
            "offset_threshold": float(model.offset_)
        }
        logger.info(
            "LOF (k=%d, c=%.2f): %d anomalies (%.2f%%) detected (threshold=%.3f)",
            k, contamination, n_anomalies, pct_anomalies, model.offset_
        )
        
    # Baseline primary label (k = 20 as standard university benchmark)
    results_df["LOF_ANOMALY"] = results_df["LOF_ANOMALY_K20"]
    results_df["LOF_SCORE"] = results_df["LOF_SCORE_K20"]
    
    if save_outputs:
        out_csv = out_dir / "lof_results.csv"
        results_df.to_csv(out_csv, index=False)
        logger.info("Saved LOF predictions to: %s", out_csv)
        
        summary_path = root / "outputs" / "evaluation_results" / "lof_summary.json"
        with open(summary_path, "w") as f:
            json.dump(experiment_metrics, f, indent=2)
            
        plot_lof_diagnostics(results_df, graph_dir)
        
    return results_df, experiment_metrics


def plot_lof_diagnostics(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate diagnostic visualizations for Local Outlier Factor experiments."""
    sns.set_theme(style="whitegrid", palette="deep")
    
    # 1. Negative Outlier Factor Distributions across k
    plt.figure(figsize=(9, 5))
    for k in NEIGHBORHOOD_SIZES:
        col = f"LOF_SCORE_K{k}"
        sns.kdeplot(df[col], label=f"k={k}", fill=True, alpha=0.15)
    plt.title("LOF Negative Outlier Score Distribution across k Neighbors", fontsize=13, weight="bold")
    plt.xlabel("Negative Outlier Factor (More Negative = Lower Density)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "lof_score_distributions.png", dpi=200)
    plt.close()
    
    # 2. Local Reachability Density Anomalies on TMIN vs. TMAX
    plt.figure(figsize=(10, 6))
    normal = df[df["LOF_ANOMALY"] == 0]
    anomalous = df[df["LOF_ANOMALY"] == 1]
    
    plt.scatter(normal["TMIN"], normal["TMAX"], c="lightgray", label="Normal Density", alpha=0.4, s=20)
    plt.scatter(anomalous["TMIN"], anomalous["TMAX"], c="royalblue", label="LOF Anomaly (k=20)", alpha=0.85, s=35, edgecolors="black", linewidths=0.5)
    
    plt.title("Local Outlier Factor Anomaly Detection on TMIN vs. TMAX Plane", fontsize=13, weight="bold")
    plt.xlabel("Minimum Temperature (°C)", fontsize=11)
    plt.ylabel("Maximum Temperature (°C)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "lof_temperature_anomalies.png", dpi=200)
    plt.close()
    
    # 3. Anomaly Count by Station
    plt.figure(figsize=(8, 5))
    station_counts = df[df["LOF_ANOMALY"] == 1]["STATION"].value_counts()
    sns.barplot(x=station_counts.index, y=station_counts.values, hue=station_counts.index, palette="Blues_r", legend=False)
    plt.title("LOF Anomalies by Weather Station (k=20)", fontsize=13, weight="bold")
    plt.xlabel("Station Identifier", fontsize=11)
    plt.ylabel("Count of Flagged Anomalies", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "lof_anomalies_by_station.png", dpi=200)
    plt.close()
    
    logger.info("Saved LOF diagnostic figures to: %s", output_dir)


if __name__ == "__main__":
    print("=" * 70)
    print("Local Outlier Factor Anomaly Detection (INT396 / SIH26073)")
    print("=" * 70)
    res_df, metrics = run_lof_experiments()
    print("\nLOF Experiment Summary:")
    for k, v in metrics.items():
        print(f" - {k}: {v['anomalies_detected']} anomalies ({v['anomaly_percentage']}%) | Offset: {v['offset_threshold']:.3f}")

