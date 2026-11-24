"""Principal Component Analysis (PCA) Visualization Module.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Projects 10-dimensional scaled meteorological feature space onto 2D orthogonal
principal component axes to visualize, explore, and interpret anomaly clusters.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_loader import get_project_root
from src.data_preprocessing import NUMERICAL_FEATURE_COLS

logger = logging.getLogger("pca_analysis")
os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".cache" / "matplotlib")
plt.switch_backend("Agg")


def fit_pca_projection(
    X_scaled: Union[pd.DataFrame, np.ndarray],
    n_components: int = 2,
    random_state: int = 42
) -> Tuple[np.ndarray, PCA, Dict[str, float]]:
    """Compute 2D PCA projection and explained variance metrics.
    
    Parameters
    ----------
    X_scaled : pd.DataFrame or np.ndarray
        Standard-scaled numerical feature matrix.
    n_components : int, default 2
        Number of orthogonal eigenvectors to retain for 2D visualization.
    random_state : int, default 42
        Reproducibility seed for SVD solver.
        
    Returns
    -------
    X_pca : np.ndarray
        Projected coordinates (N, 2).
    pca : PCA
        Fitted Scikit-learn PCA instance.
    metrics : dict
        Explained variance ratio for PC1, PC2, and cumulative sum.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    evr = pca.explained_variance_ratio_
    metrics = {
        "pc1_explained_variance": float(evr[0]),
        "pc2_explained_variance": float(evr[1]),
        "cumulative_explained_variance": float(np.sum(evr)),
        "n_features_in": int(pca.n_features_in_)
    }
    logger.info(
        "PCA Fitted: PC1=%.2f%%, PC2=%.2f%% (Cumulative=%.2f%%)",
        evr[0] * 100, evr[1] * 100, np.sum(evr) * 100
    )
    return X_pca, pca, metrics


def compute_component_loadings(
    pca: PCA,
    feature_names: List[str]
) -> pd.DataFrame:
    """Extract eigenvector feature loadings to explain physical meaning of axes."""
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1_Loading", "PC2_Loading"],
        index=feature_names
    )
    loadings["PC1_Abs"] = loadings["PC1_Loading"].abs()
    loadings["PC2_Abs"] = loadings["PC2_Loading"].abs()
    return loadings.sort_values(by="PC1_Abs", ascending=False)


def run_pca_pipeline(save_outputs: bool = True) -> Tuple[pd.DataFrame, PCA, Dict]:
    """Execute PCA projection, feature loading audit, and graph generation."""
    root = get_project_root()
    data_file = root / "outputs" / "anomaly_results" / "consensus_anomaly_results.csv"
    df = pd.read_csv(data_file, low_memory=False)
    
    scaled_feature_cols = [f"{c}_SCALED" for c in NUMERICAL_FEATURE_COLS]
    X_scaled = df[scaled_feature_cols]
    
    # Fit PCA
    X_pca, pca, metrics = fit_pca_projection(X_scaled, n_components=2)
    
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]
    
    # Feature loadings
    loadings_df = compute_component_loadings(pca, NUMERICAL_FEATURE_COLS)
    
    if save_outputs:
        out_table_dir = root / "outputs" / "comparison_tables"
        graph_dir = root / "graphs" / "pca"
        out_table_dir.mkdir(parents=True, exist_ok=True)
        graph_dir.mkdir(parents=True, exist_ok=True)
        
        # Save enriched CSV with PC coordinates
        df.to_csv(data_file, index=False)
        
        # Save loadings table
        loadings_path = out_table_dir / "pca_component_loadings.csv"
        loadings_df.to_csv(loadings_path)
        logger.info("Saved PCA loadings to: %s", loadings_path)
        
        # Save summary JSON
        summary_path = root / "outputs" / "evaluation_results" / "pca_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "variance_metrics": metrics,
                "top_pc1_drivers": loadings_df.sort_values(by="PC1_Abs", ascending=False).index[:3].tolist(),
                "top_pc2_drivers": loadings_df.sort_values(by="PC2_Abs", ascending=False).index[:3].tolist()
            }, f, indent=2)
            
        plot_pca_visualizations(df, pca, loadings_df, graph_dir)
        
    return df, pca, metrics


def plot_pca_visualizations(
    df: pd.DataFrame,
    pca: PCA,
    loadings_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Generate separate and combined publication-ready PCA scatter projections."""
    sns.set_theme(style="whitegrid")
    ev1 = pca.explained_variance_ratio_[0] * 100
    ev2 = pca.explained_variance_ratio_[1] * 100
    
    # 1. Isolation Forest in 2D PCA Space
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df[df["IF_ANOMALY"] == 0],
        x="PC1", y="PC2",
        c="#bdc3c7", label="Normal", alpha=0.35, s=20
    )
    sns.scatterplot(
        data=df[df["IF_ANOMALY"] == 1],
        x="PC1", y="PC2",
        c="#e74c3c", label="Isolation Forest Anomaly", alpha=0.85, s=40, edgecolors="black", linewidths=0.4
    )
    plt.title(f"Isolation Forest Anomalies in 2D PCA Space\n(Total Explained Variance: {ev1+ev2:.1f}%)", fontsize=12, weight="bold")
    plt.xlabel(f"Principal Component 1 ({ev1:.1f}% Variance Explained)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({ev2:.1f}% Variance Explained)", fontsize=11)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_isolation_forest.png", dpi=200)
    plt.close()
    
    # 2. Local Outlier Factor in 2D PCA Space
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df[df["LOF_ANOMALY"] == 0],
        x="PC1", y="PC2",
        c="#bdc3c7", label="Normal Density", alpha=0.35, s=20
    )
    sns.scatterplot(
        data=df[df["LOF_ANOMALY"] == 1],
        x="PC1", y="PC2",
        c="#2980b9", label="LOF Anomaly", alpha=0.85, s=40, edgecolors="black", linewidths=0.4
    )
    plt.title(f"Local Outlier Factor Anomalies in 2D PCA Space\n(Total Explained Variance: {ev1+ev2:.1f}%)", fontsize=12, weight="bold")
    plt.xlabel(f"Principal Component 1 ({ev1:.1f}% Variance Explained)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({ev2:.1f}% Variance Explained)", fontsize=11)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_local_outlier_factor.png", dpi=200)
    plt.close()
    
    # 3. Consensus vs. Method-Specific Anomalies in PCA Space
    plt.figure(figsize=(10, 8))
    palette = {
        "Normal": "#dcdde1",
        "Consensus Anomaly": "#e84118",
        "IF Only": "#f39c12",
        "LOF Only": "#0984e3"
    }
    sns.scatterplot(
        data=df,
        x="PC1", y="PC2",
        hue="CONSENSUS_LABEL",
        palette=palette,
        alpha=0.75,
        s=35
    )
    plt.title("Algorithm Consensus & Divergence in 2D PCA Space", fontsize=13, weight="bold")
    plt.xlabel(f"Principal Component 1 ({ev1:.1f}%)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({ev2:.1f}%)", fontsize=11)
    plt.legend(title="Classification")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_consensus_breakdown.png", dpi=200)
    plt.close()
    
    # 4. Weather Stations Clusters in PCA Space (Geographical/Climatic verification)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="PC1", y="PC2",
        hue="STATION",
        alpha=0.5,
        s=25
    )
    plt.title("Weather Station Climatological Dispersion in 2D PCA Space", fontsize=13, weight="bold")
    plt.xlabel(f"Principal Component 1 ({ev1:.1f}%)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({ev2:.1f}%)", fontsize=11)
    plt.legend(title="Station")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_stations_climate_dispersion.png", dpi=200)
    plt.close()
    
    logger.info("Saved PCA figures to: %s", output_dir)


if __name__ == "__main__":
    print("=" * 70)
    print("Principal Component Analysis (PCA) Projection (INT396 / SIH26073)")
    print("=" * 70)
    df_pca, pca_model, var_metrics = run_pca_pipeline()
    print("\nPCA Variance Explained:")
    print(f" - PC1: {var_metrics['pc1_explained_variance']*100:.2f}%")
    print(f" - PC2: {var_metrics['pc2_explained_variance']*100:.2f}%")
    print(f" - Cumulative (2D): {var_metrics['cumulative_explained_variance']*100:.2f}%")
    print("\nComponent Loadings (Top Features):")
    loadings = compute_component_loadings(pca_model, NUMERICAL_FEATURE_COLS)
    print(loadings[["PC1_Loading", "PC2_Loading"]].round(3))

