"""Unit tests for Isolation Forest, LOF, PCA, and comparative metrics."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.isolation_forest import train_and_predict_isolation_forest
from src.lof_detection import train_and_predict_lof
from src.pca_analysis import fit_pca_projection
from src.comparison import compute_anomaly_metrics


@pytest.fixture
def synthetic_data():
    """Generate small synthetic weather feature matrix with known outliers."""
    np.random.seed(42)
    # 200 normal observations
    normal = np.random.normal(loc=0.0, scale=1.0, size=(200, 5))
    # 10 clear extreme outliers
    outliers = np.random.uniform(low=8.0, high=12.0, size=(10, 5))
    X = np.vstack([normal, outliers])
    return X


def test_isolation_forest_execution(synthetic_data):
    """Test Isolation Forest binary output shape and decision function scores."""
    labels, scores, model = train_and_predict_isolation_forest(
        synthetic_data, contamination=0.05, n_estimators=50, random_state=42
    )
    
    assert len(labels) == len(synthetic_data)
    assert len(scores) == len(synthetic_data)
    # Binary labels should only contain 0 and 1
    assert set(np.unique(labels)).issubset({0, 1})
    # Exactly ~5% anomalies
    assert labels.sum() > 0
    # Anomalous points have lower decision scores
    assert scores[labels == 1].mean() < scores[labels == 0].mean()


def test_lof_execution(synthetic_data):
    """Test Local Outlier Factor batch execution."""
    labels, scores, model = train_and_predict_lof(
        synthetic_data, n_neighbors=15, contamination=0.05
    )
    
    assert len(labels) == len(synthetic_data)
    assert len(scores) == len(synthetic_data)
    assert set(np.unique(labels)).issubset({0, 1})
    assert labels.sum() > 0
    # Negative outlier factor should be lower (more negative) for outliers
    assert scores[labels == 1].mean() < scores[labels == 0].mean()


def test_pca_projection(synthetic_data):
    """Verify 2D PCA projection dimensionality and explained variance."""
    X_pca, pca, metrics = fit_pca_projection(synthetic_data, n_components=2)
    
    assert X_pca.shape == (len(synthetic_data), 2)
    assert 0.0 < metrics["pc1_explained_variance"] <= 1.0
    assert 0.0 < metrics["pc2_explained_variance"] <= 1.0
    assert metrics["cumulative_explained_variance"] <= 1.0


def test_comparative_metrics():
    """Verify Jaccard similarity and agreement percentage calculations."""
    y_a = np.array([0, 0, 1, 1, 0])
    y_b = np.array([0, 0, 1, 0, 1])
    
    metrics = compute_anomaly_metrics(y_a, y_b)
    
    # Common: index 2 (1 anomaly)
    assert metrics["common_anomalies"] == 1
    # Union: indices 2, 3, 4 (3 anomalies)
    assert metrics["total_union_anomalies"] == 3
    # Jaccard = 1 / 3 = 0.3333
    assert metrics["jaccard_similarity"] == pytest.approx(0.3333, abs=1e-3)
    # Agreement: index 0 (0,0), index 1 (0,0), index 2 (1,1) -> 3 out of 5 = 60%
    assert metrics["agreement_percentage"] == pytest.approx(60.0)

