"""Unit tests for Atmosphere Intelligence Flask backend and DataService."""

import pytest
import time
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app import create_app
from app.services.data_service import DataService
from app.services.ml_service import MLService


@pytest.fixture(scope="module")
def app():
    """Create Flask application fixture for testing."""
    app = create_app("testing")
    yield app


@pytest.fixture(scope="module")
def client(app):
    """Create Flask test client fixture."""
    return app.test_client()


def test_data_service_singleton():
    """Verify DataService loads real 11,688 observations and metadata."""
    ds = DataService.get_instance()
    df = ds.get_raw_consensus_dataframe()
    
    assert len(df) == 11688, f"Expected 11688 observations, got {len(df)}"
    assert df["STATION"].nunique() == 4, f"Expected 4 stations, got {df['STATION'].nunique()}"
    assert "IF_ANOMALY" in df.columns
    assert "LOF_ANOMALY" in df.columns
    assert "CONSENSUS_LABEL" in df.columns
    assert "PC1" in df.columns and "PC2" in df.columns


def test_ml_service_baseline():
    """Verify MLService delivers verified precomputed baseline metrics."""
    ml = MLService.get_instance()
    
    # Test baseline IF (c=0.03)
    if_res = ml.run_isolation_forest(contamination=0.03, n_estimators=100)
    assert if_res["results"]["total_observations"] == 11688
    assert if_res["results"]["anomalies_detected"] == 351
    assert if_res["execution"]["cached"] is True
    
    # Test baseline LOF (k=20, c=0.03)
    lof_res = ml.run_local_outlier_factor(n_neighbors=20, contamination=0.03)
    assert lof_res["results"]["total_observations"] == 11688
    assert lof_res["results"]["anomalies_detected"] == 351
    assert lof_res["execution"]["cached"] is True


def test_api_health_endpoint(client):
    """Test GET /api/health returns 200 and healthy status."""
    t0 = time.perf_counter()
    resp = client.get("/api/health")
    latency_ms = (time.perf_counter() - t0) * 1000
    
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["dataset_loaded"] is True
    assert data["observations_in_memory"] == 11688
    assert data["stations_count"] == 4
    print(f"\n[Bench] /api/health response time: {latency_ms:.2f} ms")


def test_api_overview_endpoint(client):
    """Test GET /api/overview returns real project metrics."""
    t0 = time.perf_counter()
    resp = client.get("/api/overview")
    latency_ms = (time.perf_counter() - t0) * 1000
    
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    
    data = payload["data"]
    assert data["total_observations"] == 11688
    assert data["station_count"] == 4
    assert data["isolation_forest_baseline"]["anomalies"] == 351
    assert data["lof_baseline"]["anomalies"] == 351
    assert data["consensus_metrics"]["consensus_anomalies"] == 78
    assert data["consensus_metrics"]["agreement_percentage"] == 95.33
    assert data["consensus_metrics"]["jaccard_similarity"] == 0.1250
    assert data["pca_metrics"]["cumulative_explained_variance_pct"] == 44.05
    print(f"[Bench] /api/overview response time: {latency_ms:.2f} ms")


def test_api_dataset_summary_endpoint(client):
    """Test GET /api/dataset/summary returns station coordinates and features."""
    t0 = time.perf_counter()
    resp = client.get("/api/dataset/summary")
    latency_ms = (time.perf_counter() - t0) * 1000
    
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    
    data = payload["data"]
    assert data["total_records"] == 11688
    assert len(data["stations"]) == 4
    
    # Verify NYC Central Park coordinates
    station_ids = {s["station_id"]: s for s in data["stations"]}
    assert "USW00094728" in station_ids
    assert station_ids["USW00094728"]["records_count"] == 2922
    assert abs(station_ids["USW00094728"]["latitude"] - 40.77898) < 0.01
    
    assert len(data["feature_definitions"]) >= 10
    print(f"[Bench] /api/dataset/summary response time: {latency_ms:.2f} ms")


def test_api_comparison_endpoint(client):
    """Test GET /api/comparison returns baseline and 9-cell grid."""
    t0 = time.perf_counter()
    resp = client.get("/api/comparison")
    latency_ms = (time.perf_counter() - t0) * 1000
    
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    
    data = payload["data"]
    assert data["baseline"]["consensus_anomalies"] == 78
    assert data["baseline"]["agreement_percentage"] == 95.33
    assert len(data["parameter_grid"]) == 9
    print(f"[Bench] /api/comparison response time: {latency_ms:.2f} ms")

