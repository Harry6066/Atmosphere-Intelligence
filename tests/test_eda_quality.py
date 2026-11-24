"""Tests for Atmosphere Intelligence Phase E: Exploratory Data Analysis & Data Quality.

Verifies:
- EDA distribution statistical calculations (mean, std, median, skewness, kurtosis)
- Station-wise summaries and comparative profiles
- Monthly and seasonal climatology aggregations
- Feature correlation matrix calculations and top pairs
- Raw vs processed missingness audits
- NOAA flag distributions (trace precipitation vs quality flags)
- Physical consistency checks (TMIN <= TMAX, PRCP >= 0)
- Preprocessing feature catalog and formulas
- StandardScaler before-and-after normalization metrics
- View route rendering for /eda and /data-quality
"""

import pytest
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app import create_app
from app.services.data_service import DataService


@pytest.fixture(scope="module")
def app():
    """Create Flask testing app."""
    app = create_app("testing")
    yield app


@pytest.fixture(scope="module")
def client(app):
    """Create Flask test client."""
    return app.test_client()


# ---------------------------------------------------------------------------
# Module 3: EDA Tests
# ---------------------------------------------------------------------------

def test_eda_view_renders_cleanly(client):
    """Verify GET /eda returns 200 with initial data and visualization markup."""
    resp = client.get("/eda")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Exploratory Data Analysis" in html
    assert "ATMOSPHERE_EDA_INITIAL" in html
    assert "eda-histogram-chart" in html
    assert "eda-corr-heatmap" in html
    assert "eda-monthly-chart" in html


def test_eda_distributions_calculation(client):
    """Verify /api/eda/distributions calculates accurate histogram and summary stats."""
    resp = client.get("/api/eda/distributions?feature=TMAX&bins=30")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    data = payload["data"]

    assert data["feature"] == "TMAX"
    assert data["count"] == 11688
    summary = data["summary"]
    # Check TMAX realistic physical limits
    assert 18.0 <= summary["mean"] <= 21.0
    assert 9.0 <= summary["std"] <= 12.0
    assert -25.0 <= summary["min"] <= -15.0
    assert 38.0 <= summary["max"] <= 45.0
    assert len(data["histogram"]["counts"]) == 30


def test_eda_correlations_matrix(client):
    """Verify /api/eda/correlations returns 9x9 Pearson correlation matrix."""
    resp = client.get("/api/eda/correlations")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    data = payload["data"]

    features = data["features"]
    assert len(features) == 9
    assert "TMAX" in features
    assert "TMIN" in features
    assert "PRCP" in features

    matrix = data["matrix"]
    assert len(matrix) == 9
    assert len(matrix[0]) == 9
    # Diagonal must be 1.0
    for i in range(9):
        assert abs(matrix[i][i] - 1.0) < 1e-4

    # TMAX and TMIN correlation should be strongly positive (> 0.90)
    idx_max = features.index("TMAX")
    idx_min = features.index("TMIN")
    assert matrix[idx_max][idx_min] > 0.90


def test_eda_seasonal_climatology(client):
    """Verify /api/eda/seasonal calculates 12 monthly and 4 seasonal profiles."""
    resp = client.get("/api/eda/seasonal?variable=TMAX")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    data = payload["data"]

    monthly = data["monthly"]
    assert len(monthly) == 12
    # July (month 7) should be substantially warmer than January (month 1)
    jan_mean = monthly[0]["mean"]
    jul_mean = monthly[6]["mean"]
    assert jul_mean > jan_mean + 10.0

    seasonal = data["seasonal"]
    assert len(seasonal) == 4
    # Summer (season 3) mean > Winter (season 1) mean
    summer_mean = seasonal[2]["mean"]
    winter_mean = seasonal[0]["mean"]
    assert summer_mean > winter_mean


def test_eda_station_comparison_profiles(client):
    """Verify /api/eda/station-comparison returns distinct climatology for 4 stations."""
    resp = client.get("/api/eda/station-comparison")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    stations = payload["data"]["stations"]
    assert len(stations) == 4

    stn_dict = {s["station_id"]: s for s in stations}

    # Miami should have 0 snow days and highest mean TMAX
    mia = stn_dict["USW00012839"]
    assert mia["moisture"]["snow_days_count"] == 0
    assert mia["temperature"]["mean_tmax"] > 28.0

    # Chicago should have lowest extreme minimum (< -20°C)
    chi = stn_dict["USW00094846"]
    assert chi["temperature"]["extreme_low"] <= -20.0


# ---------------------------------------------------------------------------
# Module 4: Data Quality & Preprocessing Tests
# ---------------------------------------------------------------------------

def test_data_quality_view_renders_cleanly(client):
    """Verify GET /data-quality returns 200 with physical audit and workflow markup."""
    resp = client.get("/data-quality")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Data Quality &amp; Preprocessing" in html or "Data Quality & Preprocessing" in html
    assert "Physical Consistency Sanity Checks" in html
    assert "StandardScaler" in html
    assert "ghcnd_raw_combined.csv" in html


def test_preprocessing_audit_api(client):
    """Verify /api/preprocessing/audit confirms 0 physical rule violations and 1,352 trace flags."""
    resp = client.get("/api/preprocessing/audit")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    data = payload["data"]

    dataset_audit = data["dataset_audit"]
    assert dataset_audit["raw_total_rows"] == 11688
    assert dataset_audit["processed_total_rows"] == 11688
    assert dataset_audit["duplicate_records_found"] == 0
    assert dataset_audit["station_coverage_pct"] == 100.0

    # Physical checks: all 0 violations
    for check in data["physical_consistency_checks"]:
        assert check["violations_found"] == 0
        assert check["status"] == "PASSED"

    # Trace precipitation flag
    assert data["measurement_flags"]["Total_Trace_Observations"] == 1352


def test_preprocessing_features_catalog(client):
    """Verify /api/preprocessing/features catalogs all 8 engineered features."""
    resp = client.get("/api/preprocessing/features")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    features = payload["data"]["features"]
    assert len(features) == 8

    feat_names = [f["feature"] for f in features]
    assert "TEMP_RANGE" in feat_names
    assert "TEMP_MEAN" in feat_names
    assert "TMAX_DIFF_1D" in feat_names
    assert "TMIN_DIFF_1D" in feat_names
    assert "SIN_DOY" in feat_names
    assert "COS_DOY" in feat_names
    assert "HAS_PRCP" in feat_names
    assert "HAS_SNOW" in feat_names


def test_preprocessing_scaling_comparison(client):
    """Verify /api/preprocessing/scaling verifies zero-mean and unit-variance."""
    resp = client.get("/api/preprocessing/scaling")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    table = payload["data"]["comparison_table"]
    assert len(table) == 10

    for row in table:
        # Scaled mean must be ~0.0 (+/- 0.01)
        assert abs(row["scaled_mean"]) < 0.01
        # Scaled std must be ~1.0 (+/- 0.01)
        assert abs(row["scaled_std"] - 1.0) < 0.01


def test_eda_invalid_feature_fallback(client):
    """Verify invalid feature parameter safely falls back to TMAX without crashing."""
    resp = client.get("/api/eda/distributions?feature=NON_EXISTENT_FEATURE_SQL_INJECTION")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    assert payload["data"]["feature"] == "TMAX"

