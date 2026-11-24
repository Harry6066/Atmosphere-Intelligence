"""Utility and Helper Functions for Weather Anomaly Detection System.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
"""

from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """Return the base path of the Faulty_Weather_Station_Detection directory."""
    current = Path(__file__).resolve().parent
    if current.name == "src":
        return current.parent
    return current


def load_processed_dataset(filename: str = "ghcnd_processed.csv") -> pd.DataFrame:
    """Load preprocessed weather observations dataset."""
    root = get_project_root()
    path = root / "data" / "processed" / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
    return df


def load_consensus_results(filename: str = "consensus_anomaly_results.csv") -> pd.DataFrame:
    """Load full consolidated anomaly results with IF, LOF, and PCA outputs."""
    root = get_project_root()
    path = root / "outputs" / "anomaly_results" / filename
    if not path.exists():
        raise FileNotFoundError(f"Consensus results not found at: {path}")
    df = pd.read_csv(path, low_memory=False)
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
    return df


def compute_jaccard_index(set_a: set, set_b: set) -> float:
    """Calculate the Jaccard similarity index between two discrete sets:
    J(A, B) = |A ∩ B| / |A ∪ B|
    """
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return float(intersection / union)

