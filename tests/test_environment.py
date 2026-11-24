"""Basic environment and dependency test to verify Phase 1 setup."""

def test_imports():
    """Verify that all core libraries can be imported without error."""
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.decomposition import PCA
    import scipy
    import matplotlib.pyplot as plt
    import seaborn as sns
    import streamlit as st
    import fpdf
    import pptx
    import requests

    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert sklearn.__version__ is not None

def test_directory_structure():
    """Verify that expected project directories exist."""
    import os
    from pathlib import Path

    base_path = Path(__file__).resolve().parent.parent
    expected_dirs = [
        base_path / "data" / "raw",
        base_path / "data" / "processed",
        base_path / "data" / "sample",
        base_path / "notebooks",
        base_path / "src",
        base_path / "models",
        base_path / "outputs" / "anomaly_results",
        base_path / "outputs" / "comparison_tables",
        base_path / "outputs" / "evaluation_results",
        base_path / "graphs" / "eda",
        base_path / "graphs" / "isolation_forest",
        base_path / "graphs" / "lof",
        base_path / "graphs" / "pca",
        base_path / "graphs" / "comparisons",
        base_path / "app" / "components",
        base_path / "tests",
        base_path / "docs"
    ]

    for d in expected_dirs:
        assert d.exists(), f"Directory missing: {d}"

