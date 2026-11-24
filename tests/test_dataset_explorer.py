"""Tests for Atmosphere Intelligence Phase D: Executive Overview & Dataset Explorer.

Verifies:
- Overview metrics match backend results exactly
- Station coordinates match NOAA source metadata
- Station and date filtering correctness
- Search and sorting correctness
- Pagination accuracy and edge cases
- Missing-value and quality flag summaries
- CSV export format, headers, and filtering
- Empty and invalid filter handling
"""

import pytest
from pathlib import Path
import sys
import io
import csv

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
# Module 1: Overview Tests
# ---------------------------------------------------------------------------

def test_overview_metrics_match_backend_truth(client):
    """Verify overview metrics match verified experimental benchmarks."""
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    data = payload["data"]

    # Verify exact experimental metrics
    assert data["total_observations"] == 11688
    assert data["station_count"] == 4
    assert data["isolation_forest_baseline"]["anomalies"] == 351
    assert data["isolation_forest_baseline"]["contamination"] == 0.03
    assert data["lof_baseline"]["anomalies"] == 351
    assert data["lof_baseline"]["n_neighbors"] == 20
    assert data["consensus_metrics"]["consensus_anomalies"] == 78
    assert data["consensus_metrics"]["agreement_percentage"] == 95.33
    assert data["consensus_metrics"]["jaccard_similarity"] == 0.1250
    assert data["pca_metrics"]["cumulative_explained_variance_pct"] == 44.05


def test_overview_view_renders_cleanly(client):
    """Verify GET / renders overview page with all 8 pipeline stages and metrics."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "ATMOSPHERE" in html
    assert "SIH26073" in html
    assert "11,688" in html or "11688" in html
    assert "351" in html
    assert "78" in html
    assert "95.33%" in html
    assert "0.1250" in html
    assert "Unsupervised Scientific Detection Pipeline" in html
    assert "overview-consensus-chart" in html


# ---------------------------------------------------------------------------
# Module 2: Station Coordinates and Metadata Tests
# ---------------------------------------------------------------------------

def test_station_coordinates_match_noaa_metadata(client):
    """Verify all 4 NOAA stations have exact coordinates and metadata."""
    resp = client.get("/api/dataset/stations")
    assert resp.status_code == 200
    stations = resp.get_json()["data"]
    assert len(stations) == 4

    stn_map = {s["station_id"]: s for s in stations}
    
    # Check NYC Central Park
    nyc = stn_map["USW00094728"]
    assert "CENTRAL PARK" in nyc["name"].upper()
    assert round(nyc["latitude"], 2) == 40.78
    assert round(nyc["longitude"], 2) == -73.97
    assert nyc["elevation_m"] == 42.7
    assert nyc["records_count"] == 2922

    # Check Chicago O'Hare
    chi = stn_map["USW00094846"]
    assert "CHICAGO" in chi["name"].upper()
    assert round(chi["latitude"], 2) == 41.96
    assert round(chi["longitude"], 2) == -87.93
    assert chi["elevation_m"] == 204.8
    assert chi["records_count"] == 2922

    # Check Seattle-Tacoma
    sea = stn_map["USW00024233"]
    assert "SEATTLE" in sea["name"].upper()
    assert round(sea["latitude"], 2) == 47.44
    assert round(sea["longitude"], 2) == -122.31
    assert sea["elevation_m"] == 112.5
    assert sea["records_count"] == 2922

    # Check Miami
    mia = stn_map["USW00012839"]
    assert "MIAMI" in mia["name"].upper()
    assert round(mia["latitude"], 2) == 25.79
    assert round(mia["longitude"], 2) == -80.32
    assert mia["elevation_m"] == 1.4
    assert mia["records_count"] == 2922


# ---------------------------------------------------------------------------
# Module 2: Pagination and Filtering Tests
# ---------------------------------------------------------------------------

def test_dataset_observations_pagination_correctness(client):
    """Verify server-side pagination slices records accurately without leaks."""
    resp = client.get("/api/dataset/observations?page=1&page_size=20")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    assert len(payload["data"]) == 20
    assert payload["pagination"]["total_records"] == 11688
    assert payload["pagination"]["total_pages"] == 585
    assert payload["pagination"]["page"] == 1

    # Second page check
    resp2 = client.get("/api/dataset/observations?page=2&page_size=20")
    assert resp2.status_code == 200
    p2_data = resp2.get_json()["data"]
    assert len(p2_data) == 20
    # Ensure page 1 and page 2 do not overlap
    assert payload["data"][0]["date"] != p2_data[0]["date"] or payload["data"][0]["station"] != p2_data[0]["station"]


def test_station_filtering(client):
    """Verify filtering by single station returns exactly 2,922 records."""
    resp = client.get("/api/dataset/records?station=USW00094728&page_size=10")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["pagination"]["total_records"] == 2922
    for r in payload["data"]:
        assert r["station"] == "USW00094728"


def test_date_range_filtering(client):
    """Verify filtering by date range isolates specific periods (e.g., Polar Vortex)."""
    # 2019 Polar Vortex period: Jan 25 to Feb 02, 2019 (9 days across 4 stations = 36 records)
    resp = client.get("/api/dataset/records?start_date=2019-01-25&end_date=2019-02-02")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["pagination"]["total_records"] == 36
    for r in payload["data"]:
        assert "2019-01-25" <= r["date"] <= "2019-02-02"


def test_anomaly_status_filtering(client):
    """Verify consensus outlier filter returns exactly 78 records."""
    resp = client.get("/api/dataset/records?anomaly_type=consensus")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["pagination"]["total_records"] == 78
    for r in payload["data"]:
        assert r["consensus_label"] == "Consensus Anomaly"


def test_search_and_sorting(client):
    """Verify text search and sorting by TMAX descending."""
    resp = client.get("/api/dataset/records?search=Miami&sort_col=tmax&sort_dir=desc&page_size=5")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["pagination"]["total_records"] > 0
    tmax_values = [r["tmax"] for r in payload["data"]]
    assert tmax_values == sorted(tmax_values, reverse=True)


# ---------------------------------------------------------------------------
# Module 2: Data Quality and Export Tests
# ---------------------------------------------------------------------------

def test_quality_and_flag_stats(client):
    """Verify missingness and quality flag statistics."""
    resp = client.get("/api/dataset/quality-stats")
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["total_observations"] == 11688
    # 0.0% missing for core features
    assert data["missingness_percentages"]["TMAX"] == 0.0
    assert data["missingness_percentages"]["PRCP"] == 0.0
    # Zero physical rule violations
    assert data["physical_rule_violations"]["inverted_temp_count"] == 0
    assert data["physical_rule_violations"]["negative_prcp_count"] == 0
    assert data["physical_rule_violations"]["official_qc_flag_count"] == 0


def test_csv_export_format_and_headers(client):
    """Verify CSV export preserves columns, headers, and filtered rows."""
    resp = client.get("/api/dataset/export?station=USW00094728&year=2020")
    assert resp.status_code == 200
    assert resp.mimetype == "text/csv"
    assert "attachment; filename=" in resp.headers.get("Content-Disposition", "")

    csv_text = resp.get_data(as_text=True)
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    header = rows[0]

    assert "DATE" in header
    assert "STATION" in header
    assert "TMAX" in header
    assert "TMIN" in header
    assert "PRCP" in header
    assert "CONSENSUS_LABEL" in header

    # Leap year 2020 has 366 days + 1 header = 367 rows
    assert len(rows) == 367


def test_empty_result_and_invalid_filters_graceful(client):
    """Verify non-matching or invalid filters return empty structure safely without crashing."""
    # Future year with zero records
    resp = client.get("/api/dataset/records?year=2099")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    assert len(payload["data"]) == 0
    assert payload["pagination"]["total_records"] == 0
    assert payload["summary"]["filtered_count"] == 0

    # Invalid sort column safely falls back
    resp_invalid_sort = client.get("/api/dataset/records?sort_col=malicious_sql&sort_dir=invalid")
    assert resp_invalid_sort.status_code == 200
    assert resp_invalid_sort.get_json()["success"] is True
