"""Comparative Analysis Module between Isolation Forest and Local Outlier Factor.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Computes intersection consensus, algorithm-specific divergences, Jaccard similarity,
cross-parameter stability matrices, and outputs publication-quality comparison charts.
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

logger = logging.getLogger("comparison")
os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".cache" / "matplotlib")
plt.switch_backend("Agg")


def compute_anomaly_metrics(
    y_true_or_a: np.ndarray,
    y_b: np.ndarray
) -> Dict[str, Union[int, float]]:
    """Compute consensus, algorithm-specific set sizes, agreement, and Jaccard similarity.
    
    Parameters
    ----------
    y_true_or_a : np.ndarray
        Binary indicators (1=anomaly, 0=normal) for Algorithm A (e.g. IF).
    y_b : np.ndarray
        Binary indicators (1=anomaly, 0=normal) for Algorithm B (e.g. LOF).
        
    Returns
    -------
    dict
        Comparative metrics breakdown.
    """
    total = len(y_true_or_a)
    set_a = set(np.where(y_true_or_a == 1)[0])
    set_b = set(np.where(y_b == 1)[0])
    
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    only_a = set_a - set_b
    only_b = set_b - set_a
    
    # Agreement: both flag anomaly OR both flag normal
    both_normal = total - len(union)
    total_agreed = len(intersection) + both_normal
    agreement_pct = (total_agreed / total) * 100.0
    
    jaccard = (len(intersection) / len(union)) if len(union) > 0 else 0.0
    
    return {
        "total_observations": total,
        "anomalies_a": len(set_a),
        "anomalies_b": len(set_b),
        "common_anomalies": len(intersection),
        "only_a_anomalies": len(only_a),
        "only_b_anomalies": len(only_b),
        "total_union_anomalies": len(union),
        "agreement_percentage": round(agreement_pct, 2),
        "jaccard_similarity": round(jaccard, 4)
    }


def categorize_detection_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """Label each observation by consensus category:
    - 'Consensus Anomaly': Detected by both IF and LOF.
    - 'IF Only': Global tree isolation anomaly only.
    - 'LOF Only': Local density anomaly only.
    - 'Normal': Not flagged by either model.
    """
    data = df.copy()
    
    conditions = [
        (data["IF_ANOMALY"] == 1) & (data["LOF_ANOMALY"] == 1),
        (data["IF_ANOMALY"] == 1) & (data["LOF_ANOMALY"] == 0),
        (data["IF_ANOMALY"] == 0) & (data["LOF_ANOMALY"] == 1),
    ]
    choices = ["Consensus Anomaly", "IF Only", "LOF Only"]
    data["CONSENSUS_LABEL"] = np.select(conditions, choices, default="Normal")
    
    return data


def run_comparative_pipeline(save_outputs: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Execute complete comparative evaluation between IF and LOF."""
    root = get_project_root()
    if_file = root / "outputs" / "anomaly_results" / "isolation_forest_results.csv"
    lof_file = root / "outputs" / "anomaly_results" / "lof_results.csv"
    
    df_if = pd.read_csv(if_file, low_memory=False)
    df_lof = pd.read_csv(lof_file, low_memory=False)
    
    # Merge model outputs
    combined = df_if.copy()
    for col in df_lof.columns:
        if col.startswith("LOF_"):
            combined[col] = df_lof[col]
            
    combined = categorize_detection_consensus(combined)
    
    # 1. Baseline Primary Comparison (IF c=0.03 vs LOF k=20)
    baseline_metrics = compute_anomaly_metrics(
        combined["IF_ANOMALY"].values, combined["LOF_ANOMALY"].values
    )
    logger.info("Baseline IF vs LOF Metrics: %s", baseline_metrics)
    
    # 2. Cross-Parameter Sensitivity Grid
    if_cols = ["IF_ANOMALY_C01", "IF_ANOMALY_C03", "IF_ANOMALY_C05"]
    lof_cols = ["LOF_ANOMALY_K10", "LOF_ANOMALY_K20", "LOF_ANOMALY_K30"]
    
    grid_records = []
    for if_c in if_cols:
        c_val = int(if_c.split("_C")[1]) / 100.0
        for lof_k in lof_cols:
            k_val = int(lof_k.split("_K")[1])
            m = compute_anomaly_metrics(combined[if_c].values, combined[lof_k].values)
            grid_records.append({
                "IF_Contamination": c_val,
                "LOF_Neighbors": k_val,
                "IF_Anomalies": m["anomalies_a"],
                "LOF_Anomalies": m["anomalies_b"],
                "Common_Consensus": m["common_anomalies"],
                "IF_Only": m["only_a_anomalies"],
                "LOF_Only": m["only_b_anomalies"],
                "Agreement_Pct": m["agreement_percentage"],
                "Jaccard_Similarity": m["jaccard_similarity"]
            })
            
    df_grid = pd.DataFrame(grid_records)
    
    if save_outputs:
        out_table_dir = root / "outputs" / "comparison_tables"
        graph_dir = root / "graphs" / "comparisons"
        out_table_dir.mkdir(parents=True, exist_ok=True)
        graph_dir.mkdir(parents=True, exist_ok=True)
        
        # Save enriched consolidated dataset
        consolidated_path = root / "outputs" / "anomaly_results" / "consensus_anomaly_results.csv"
        combined.to_csv(consolidated_path, index=False)
        logger.info("Saved consensus results to: %s", consolidated_path)
        
        # Save cross-parameter comparison table
        table_path = out_table_dir / "model_comparison_matrix.csv"
        df_grid.to_csv(table_path, index=False)
        logger.info("Saved parameter comparison table to: %s", table_path)
        
        # Save baseline metrics JSON
        summary_path = root / "outputs" / "evaluation_results" / "comparative_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "baseline_comparison": baseline_metrics,
                "consensus_counts": combined["CONSENSUS_LABEL"].value_counts().to_dict()
            }, f, indent=2)
            
        plot_comparative_visualizations(combined, df_grid, graph_dir)
        
    return combined, df_grid, baseline_metrics


def plot_comparative_visualizations(
    df: pd.DataFrame,
    df_grid: pd.DataFrame,
    output_dir: Path
) -> None:
    """Generate comparative visualization plots."""
    sns.set_theme(style="whitegrid")
    
    # 1. Consensus Category Breakdown
    plt.figure(figsize=(8, 5))
    counts = df["CONSENSUS_LABEL"].value_counts()
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]
    sns.barplot(x=counts.index, y=counts.values, palette="Set2")
    for i, v in enumerate(counts.values):
        plt.text(i, v + 50, f"{v:,}", ha="center", fontweight="bold", fontsize=10)
    plt.title("Observation Classification by Detection Consensus", fontsize=13, weight="bold")
    plt.xlabel("Consensus Category", fontsize=11)
    plt.ylabel("Observation Count", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "consensus_category_breakdown.png", dpi=200)
    plt.close()
    
    # 2. Jaccard Similarity Heatmap across Parameter Grid
    plt.figure(figsize=(7, 6))
    pivot_jaccard = df_grid.pivot(
        index="IF_Contamination",
        columns="LOF_Neighbors",
        values="Jaccard_Similarity"
    )
    sns.heatmap(pivot_jaccard, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={"label": "Jaccard Index"})
    plt.title("Jaccard Similarity: Isolation Forest vs. LOF Across Parameters", fontsize=12, weight="bold")
    plt.xlabel("LOF Neighborhood Size (k)", fontsize=11)
    plt.ylabel("Isolation Forest Contamination Rate (c)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_jaccard_heatmap.png", dpi=200)
    plt.close()
    
    # 3. Weather Distribution for Consensus vs. Method-Specific Anomalies
    plt.figure(figsize=(10, 6))
    palette = {
        "Normal": "#bdc3c7",
        "Consensus Anomaly": "#e74c3c",
        "IF Only": "#e67e22",
        "LOF Only": "#2980b9"
    }
    sns.scatterplot(
        data=df,
        x="TMAX",
        y="TEMP_RANGE",
        hue="CONSENSUS_LABEL",
        palette=palette,
        alpha=0.75,
        s=30
    )
    plt.title("Meteorological Feature Space by Consensus Category", fontsize=13, weight="bold")
    plt.xlabel("TMAX (°C)", fontsize=11)
    plt.ylabel("Diurnal Temperature Range (TMAX - TMIN °C)", fontsize=11)
    plt.legend(title="Detection Status")
    plt.tight_layout()
    plt.savefig(output_dir / "consensus_scatter_weather_space.png", dpi=200)
    plt.close()
    
    logger.info("Saved comparison graphs to: %s", output_dir)


if __name__ == "__main__":
    print("=" * 70)
    print("Isolation Forest vs. LOF Comparative Analysis (INT396 / SIH26073)")
    print("=" * 70)
    df_combined, df_grid, baseline = run_comparative_pipeline()
    print("\nBaseline Comparison (IF c=0.03 vs LOF k=20):")
    for k, v in baseline.items():
        print(f" - {k}: {v}")
    print("\nParameter Sensitivity Grid:")
    print(df_grid.to_string(index=False))

