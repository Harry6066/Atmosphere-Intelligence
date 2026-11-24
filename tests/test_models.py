"""Tests for Atmosphere Intelligence Phase F: Isolation Forest and Local Outlier Factor Models.

Verifies:
- Module 5 view rendering (/isolation-forest)
- Module 6 view rendering (/local-outlier-factor and /lof)
- Isolation Forest baseline results and score distribution
- Isolation Forest contamination sensitivity (0.01, 0.03, 0.05) and nested subset
- Isolation Forest dynamic execution and in-memory caching
- Isolation Forest parameter boundary validation (contamination, n_estimators, features)
- LOF baseline results, negative outlier factors, and offset threshold (-1.3632)
- LOF neighborhood sensitivity (k = 10, 20, 30) and multi-scale core persistence (226)
- LOF dynamic batch execution (novelty=False) and in-memory caching
- LOF parameter boundary validation (n_neighbors, contamination, metric)
- Single observation inspection endpoint (/api/observation/inspect) with Blizzard 2016 verification
"""

import pytest
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


# -----------------------------------------------------------------------------
# Module 5: Isolation Forest Tests
# -----------------------------------------------------------------------------

def test_isolation_forest_view_renders_cleanly(client):
    """Verify GET /isolation-forest returns 200 with complete scientific markup."""
    resp = client.get("/isolation-forest")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Isolation Forest Anomaly Detection" in html
    assert "ATMOSPHERE_IF_INITIAL" in html
    assert "if-score-hist-chart" in html
    assert "if-station-bar-chart" in html
    assert "sensitivity-section" in html
    assert "observation-modal" in html


def test_isolation_forest_results_api_baseline(client):
    """Verify GET /api/isolation-forest/results delivers 351 baseline anomalies and 30 histogram bins."""
    resp = client.get("/api/isolation-forest/results?contamination=0.03&n_estimators=100")
    assert resp.status_code == 200
    data = resp.get_json()["data"]

    assert data["summary"]["total_observations"] == 11688
    assert data["summary"]["anomalies_detected"] == 351
    assert data["summary"]["normal_observations"] == 11337
    assert data["summary"]["anomaly_percentage"] == 3.0
    assert data["summary"]["score_min"] < 0
    assert len(data["histogram"]["bin_centers"]) == 30
    assert len(data["station_distribution"]) == 4

    # Chicago should lead IF anomalies (227)
    chicago_entry = next(s for s in data["station_distribution"] if s["station"] == "USW00094846")
    assert chicago_entry["anomalies"] == 227
    assert len(data["top_anomalies"]) == 25


def test_isolation_forest_sensitivity_api(client):
    """Verify GET /api/isolation-forest/sensitivity returns 117, 351, 585 anomalies."""
    resp = client.get("/api/isolation-forest/sensitivity")
    assert resp.status_code == 200
    data = resp.get_json()["data"]

    cfgs = {c["contamination"]: c["anomalies"] for c in data["configurations"]}
    assert cfgs[0.01] == 117
    assert cfgs[0.03] == 351
    assert cfgs[0.05] == 585

    assert data["overlap_analysis"]["common_all_three"] == 117
    assert data["overlap_analysis"]["added_at_c03"] == 234
    assert data["overlap_analysis"]["added_at_c05"] == 234
    assert data["overlap_analysis"]["strict_nested_subset"] is True


def test_isolation_forest_dynamic_run_and_cache(client):
    """Verify POST /api/isolation-forest/run executes dynamic model and caches result."""
    payload = {
        "contamination": 0.02,
        "n_estimators": 50,
        "random_state": 42
    }
    resp1 = client.post("/api/isolation-forest/run", json=payload)
    assert resp1.status_code == 200
    data1 = resp1.get_json()["data"]
    assert 220 <= data1["results"]["anomalies_detected"] <= 245
    assert data1["execution"]["duration_ms"] > 0

    # Second call must hit dynamic in-memory cache
    resp2 = client.post("/api/isolation-forest/run", json=payload)
    assert resp2.status_code == 200
    data2 = resp2.get_json()["data"]
    assert data2["execution"]["cached"] is True


def test_isolation_forest_parameter_validation(client):
    """Verify invalid Isolation Forest parameters are rejected with 400."""
    # Out of range contamination
    resp = client.post("/api/isolation-forest/run", json={"contamination": 0.50})
    assert resp.status_code == 400
    assert "Contamination must be between" in resp.get_json()["error"]

    # Out of range estimators
    resp = client.post("/api/isolation-forest/run", json={"n_estimators": 5})
    assert resp.status_code == 400
    assert "n_estimators must be between" in resp.get_json()["error"]

    # Invalid feature
    resp = client.post("/api/isolation-forest/run", json={"features": ["NON_EXISTENT_FEAT"]})
    assert resp.status_code == 400
    assert "Invalid feature" in resp.get_json()["error"]


# -----------------------------------------------------------------------------
# Module 6: Local Outlier Factor Tests
# -----------------------------------------------------------------------------

def test_lof_view_renders_cleanly(client):
    """Verify GET /local-outlier-factor and GET /lof both return 200 with complete markup."""
    for path in ["/local-outlier-factor", "/lof"]:
        resp = client.get(path)
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Local Outlier Factor (LOF)" in html
        assert "ATMOSPHERE_LOF_INITIAL" in html
        assert "lof-score-hist-chart" in html
        assert "Decision Offset Threshold" in html
        assert "observation-modal" in html


def test_lof_results_api_baseline(client):
    """Verify GET /api/lof/results delivers 351 baseline anomalies and offset -1.3632."""
    resp = client.get("/api/lof/results?n_neighbors=20&contamination=0.03")
    assert resp.status_code == 200
    data = resp.get_json()["data"]

    assert data["summary"]["total_observations"] == 11688
    assert data["summary"]["anomalies_detected"] == 351
    assert data["summary"]["normal_observations"] == 11337
    assert data["summary"]["anomaly_percentage"] == 3.0
    assert round(data["summary"]["offset_threshold"], 2) == -1.36

    # Verify Miami has 87 anomalies under LOF (compared to 7 under IF)
    miami_entry = next(s for s in data["station_distribution"] if s["station"] == "USW00012839")
    assert miami_entry["anomalies"] == 87
    assert len(data["top_anomalies"]) == 25


def test_lof_sensitivity_api(client):
    """Verify GET /api/lof/sensitivity returns 226 persistent core anomalies across k in {10,20,30}."""
    resp = client.get("/api/lof/sensitivity")
    assert resp.status_code == 200
    data = resp.get_json()["data"]

    assert len(data["configurations"]) == 3
    for cfg in data["configurations"]:
        assert cfg["anomalies"] == 351
        assert cfg["pct"] == 3.0

    overlap = data["overlap_analysis"]
    assert overlap["common_all_three"] == 226
    assert overlap["common_pct"] == 64.39
    assert overlap["unique_k10"] == 89
    assert overlap["unique_k20"] == 14
    assert overlap["unique_k30"] == 42


def test_lof_dynamic_run_and_cache(client):
    """Verify POST /api/lof/run executes batch LOF (novelty=False) and caches."""
    payload = {
        "n_neighbors": 15,
        "contamination": 0.03,
        "metric": "minkowski"
    }
    resp1 = client.post("/api/lof/run", json=payload)
    assert resp1.status_code == 200
    data1 = resp1.get_json()["data"]
    assert data1["results"]["anomalies_detected"] == 351
    assert data1["execution"]["duration_ms"] > 0

    # Second call should return cached result
    resp2 = client.post("/api/lof/run", json=payload)
    assert resp2.status_code == 200
    data2 = resp2.get_json()["data"]
    assert data2["execution"]["cached"] is True


def test_lof_parameter_validation(client):
    """Verify invalid LOF parameters are rejected with 400."""
    # Out of range neighbors
    resp = client.post("/api/lof/run", json={"n_neighbors": 2})
    assert resp.status_code == 400
    assert "n_neighbors must be between" in resp.get_json()["error"]

    # Invalid metric
    resp = client.post("/api/lof/run", json={"metric": "invalid_metric"})
    assert resp.status_code == 400
    assert "Invalid metric" in resp.get_json()["error"]


# -----------------------------------------------------------------------------
# Observation Inspection Modal Endpoint Tests
# -----------------------------------------------------------------------------

def test_observation_inspection_endpoint(client):
    """Verify GET /api/observation/inspect returns detailed profile for 2016 Blizzard."""
    # Central Park on 2016-01-23 (Winter Storm Jonas record blizzard)
    resp = client.get("/api/observation/inspect?station=USW00094728&date=2016-01-23")
    assert resp.status_code == 200
    ob = resp.get_json()["data"]

    assert ob["station"] == "USW00094728"
    assert ob["date"] == "2016-01-23"
    assert ob["observations"]["tmax"] == -2.7
    assert ob["observations"]["snow"] == 693.0
    assert ob["models"]["isolation_forest"]["prediction"] == "Anomaly"
    assert ob["models"]["local_outlier_factor"]["prediction"] == "Anomaly"
    assert ob["models"]["consensus"]["is_consensus"] is True
    assert ob["quality_flags"]["passed_physical_checks"] is True
    assert "scientific_disclaimer" in ob


def test_observation_inspection_error_handling(client):
    """Verify missing and non-existent parameters in /api/observation/inspect."""
    # Missing date
    resp = client.get("/api/observation/inspect?station=USW00094728")
    assert resp.status_code == 400

    # Non-existent observation date
    resp = client.get("/api/observation/inspect?station=USW00094728&date=1900-01-01")
    assert resp.status_code == 404

