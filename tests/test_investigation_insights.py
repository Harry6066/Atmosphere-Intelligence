"""Unit and integration tests for Phase H: Anomaly Investigation and Scientific Insights.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Tests Module 09 (/anomaly-investigation) and Module 10 (/scientific-insights),
verifying analytical services, REST API contracts, filter correctness, denominator
alignment, CSV streaming, and regression safety.
"""

import io
import csv
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
    """Create Flask application in testing mode."""
    app = create_app("testing")
    yield app


@pytest.fixture(scope="module")
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture(scope="module")
def data_service():
    """Retrieve singleton DataService."""
    return DataService.get_instance()


# =============================================================================
# 1. INVESTIGATION SUMMARY & DATA SERVICE UNIT TESTS
# =============================================================================

def test_investigation_summary_metrics(data_service):
    """Verify exact counts and rates in investigation summary."""
    summary = data_service.get_investigation_summary()
    assert summary["total_observations"] == 11688
    assert summary["if_anomalies"] == 351
    assert summary["if_rate_pct"] == 3.0
    assert summary["lof_anomalies"] == 351
    assert summary["lof_rate_pct"] == 3.0
    assert summary["consensus_anomalies"] == 78
    assert summary["consensus_rate_pct"] == 0.67
    assert summary["if_only_anomalies"] == 273
    assert summary["lof_only_anomalies"] == 273
    assert summary["normal_observations"] == 11064
    assert summary["total_union_anomalies"] == 624
    assert summary["extreme_events_count"] == 4
    assert summary["top_suspicious_count"] == 20
    assert summary["stations_count"] == 4


def test_historical_extreme_events_audit(data_service):
    """Verify ground-truth historical climate events data structure and contents."""
    events = data_service.get_historical_extreme_events()
    assert len(events) == 4

    names = [e["event_name"] for e in events]
    assert any("Polar Vortex" in n for n in names)
    assert any("Heat Dome" in n for n in names)
    assert any("Hurricane Ida" in n for n in names)
    assert any("Blizzard" in n for n in names)

    for ev in events:
        assert ev["days"] > 0
        assert "station_id" in ev
        assert "scientific_classification" in ev
        assert ev["scientific_classification"] == "Legitimate Extreme Weather"
        assert "NOT Sensor Fault" in ev["assessment"]


def test_top_suspicious_observations_ranking(data_service):
    """Verify severity index ranking on top suspicious observations."""
    top_obs = data_service.get_top_suspicious_observations(limit=10)
    assert len(top_obs) == 10

    # Ensure strictly sorted by severity index descending
    severities = [obs["severity_index"] for obs in top_obs]
    assert severities == sorted(severities, reverse=True)
    assert severities[0] > 0.5  # Top observation severity > 0.5


# =============================================================================
# 2. OBSERVATION DETAIL, PERCENTILES & CONTEXT WINDOW
# =============================================================================

def test_observation_detail_chicago_polar_vortex(data_service):
    """Verify single observation inspector for Chicago Polar Vortex record low."""
    detail = data_service.get_observation_detail("USW00094846", "2019-01-30")
    assert detail is not None
    obs = detail["observation"]

    assert obs["station_id"] == "USW00094846"
    assert obs["date"] == "2019-01-30"
    assert obs["tmin"] == -30.5  # Verified record low
    assert obs["tmax"] == -23.2
    assert obs["if_anomaly"] == 1
    assert obs["lof_anomaly"] == 1
    assert obs["consensus_label"] == "Consensus Anomaly"
    assert obs["investigation_status"] == "Possible Genuine Meteorological Extreme"

    # Context window: exactly 7 days (D-3 to D+3)
    ctx = detail["context_window"]
    assert len(ctx) == 7
    target_items = [c for c in ctx if c["is_target"]]
    assert len(target_items) == 1
    assert target_items[0]["date"] == "2019-01-30"

    # Station percentiles: TMIN should be 0th percentile (record low)
    pct = detail["percentiles"]
    assert "TMIN" in pct
    assert pct["TMIN"]["percentile_rank"] == 0.0
    assert pct["TMIN"]["min"] == -30.5


def test_observation_detail_nonexistent(data_service):
    """Verify None returned for non-existent station or date."""
    assert data_service.get_observation_detail("INVALID_STATION", "2019-01-30") is None
    assert data_service.get_observation_detail("USW00094846", "1900-01-01") is None


# =============================================================================
# 3. TIMELINE AGGREGATION
# =============================================================================

def test_investigation_timeline_monthly_coverage(data_service):
    """Verify monthly aggregation covers all 96 months (8 years x 12 months)."""
    timeline = data_service.get_investigation_timeline()
    months = timeline["months"]
    assert len(months) == 96
    assert months[0] == "2016-01"
    assert months[-1] == "2023-12"

    # Sum of monthly IF anomalies across all stations should equal 351
    assert sum(timeline["if_anomalies"]) == 351
    assert sum(timeline["lof_anomalies"]) == 351
    assert sum(timeline["consensus_anomalies"]) == 78


def test_investigation_timeline_station_filter(data_service):
    """Verify timeline filtered to Chicago returns Chicago's 227 IF anomalies."""
    timeline = data_service.get_investigation_timeline(station="USW00094846")
    assert len(timeline["months"]) == 96
    assert sum(timeline["if_anomalies"]) == 227
    assert sum(timeline["lof_anomalies"]) == 177
    assert sum(timeline["consensus_anomalies"]) == 58


# =============================================================================
# 4. SCIENTIFIC INSIGHTS & DENOMINATOR ALIGNMENT
# =============================================================================

def test_station_bias_denominators(data_service):
    """Verify that every station has exactly 2,922 observations and rates sum properly."""
    bias = data_service.get_station_bias_insights()
    stations = bias["stations"]
    assert len(stations) == 4

    total_pool = sum(s["total_records"] for s in stations)
    assert total_pool == 11688

    for s in stations:
        assert s["total_records"] == 2922
        assert f"/ 2,922" in s["if_fraction_str"]
        assert f"/ 2,922" in s["lof_fraction_str"]
        assert f"/ 2,922" in s["consensus_fraction_str"]

    # Verify Chicago has 227 IF anomalies (7.77%) and Miami has 7 (0.24%)
    chi = next(s for s in stations if s["station_id"] == "USW00094846")
    mia = next(s for s in stations if s["station_id"] == "USW00012839")
    assert chi["if_anomalies"] == 227
    assert chi["if_rate_pct"] == 7.77
    assert mia["if_anomalies"] == 7
    assert mia["if_rate_pct"] == 0.24


def test_seasonal_bias_denominators(data_service):
    """Verify seasonal bias counts and exact denominators across calendar seasons."""
    bias = data_service.get_seasonal_bias_insights()
    seasons = bias["seasons"]
    assert len(seasons) == 4

    total_pool = sum(s["total_records"] for s in seasons)
    assert total_pool == 11688

    s_map = {s["season_name"]: s for s in seasons}
    # Winter: 2,888 records, 279 IF (9.66%)
    assert s_map["Winter"]["total_records"] == 2888
    assert s_map["Winter"]["if_anomalies"] == 279
    assert s_map["Winter"]["if_rate_pct"] == 9.66
    assert "/ 2,888" in s_map["Winter"]["if_fraction_str"]

    # Summer: 2,944 records, 5 IF (0.17%)
    assert s_map["Summer"]["total_records"] == 2944
    assert s_map["Summer"]["if_anomalies"] == 5
    assert s_map["Summer"]["if_rate_pct"] == 0.17
    assert "/ 2,944" in s_map["Summer"]["if_fraction_str"]


def test_scientific_limitations_catalog(data_service):
    """Verify complete catalog of all 9 required methodological limitations."""
    limitations = data_service.get_scientific_limitations()
    assert len(limitations) == 9

    ids = [lim["id"] for lim in limitations]
    assert ids == list(range(1, 10))

    titles = [lim["title"] for lim in limitations]
    assert any("Ground-Truth" in t for t in titles)
    assert any("Geographical Sample" in t for t in titles)
    assert any("Continentality" in t for t in titles)
    assert any("LOF" in t for t in titles)
    assert any("StandardScaler" in t for t in titles)
    assert any("PCA" in t for t in titles)


def test_ethical_guidelines(data_service):
    """Verify 6 core ethical principles for AI meteorological quality control."""
    guidelines = data_service.get_ethical_guidelines()
    assert len(guidelines) == 6
    assert all("principle" in g and "rule" in g for g in guidelines)


def test_methodology_workflow_10_steps(data_service):
    """Verify 10-step reproducibility workflow metadata."""
    wf = data_service.get_methodology_workflow()
    steps = wf["steps"]
    assert len(steps) == 10
    assert steps[0]["step"] == 1
    assert steps[-1]["step"] == 10
    assert wf["reproducibility"]["random_state"] == 42


# =============================================================================
# 5. REST API ENDPOINTS CONTRACT TESTS
# =============================================================================

def test_api_investigation_summary(client):
    """Verify GET /api/investigation/summary returns 200 with structured metrics."""
    resp = client.get("/api/investigation/summary")
    assert resp.status_code == 200
    data = resp.json["data"]
    assert data["total_observations"] == 11688
    assert data["if_anomalies"] == 351
    assert data["consensus_anomalies"] == 78


def test_api_investigation_observations_pagination(client):
    """Verify GET /api/investigation/observations pagination and filtering."""
    resp = client.get("/api/investigation/observations?page=1&per_page=10&classification=consensus")
    assert resp.status_code == 200
    data = resp.json["data"]
    assert len(data["items"]) == 10
    assert data["total"] == 78
    assert data["pages"] == 8
    assert all(it["consensus_label"] == "Consensus Anomaly" for it in data["items"])


def test_api_investigation_observation_detail(client):
    """Verify GET /api/investigation/observation/<station>/<date>."""
    resp = client.get("/api/investigation/observation/USW00094728/2021-09-01")
    assert resp.status_code == 200
    data = resp.json["data"]
    assert data["observation"]["station_id"] == "USW00094728"
    assert data["observation"]["date"] == "2021-09-01"
    assert data["observation"]["prcp"] > 70.0  # Hurricane Ida downpour


def test_api_investigation_observation_invalid_input(client):
    """Verify 400 for missing params and 404 for nonexistent observation."""
    resp_bad = client.get("/api/investigation/observation")
    assert resp_bad.status_code == 400

    resp_notfound = client.get("/api/investigation/observation/USW00099999/2099-01-01")
    assert resp_notfound.status_code == 404


def test_api_investigation_export_csv(client):
    """Verify GET /api/investigation/export streams valid CSV file."""
    resp = client.get("/api/investigation/export?classification=consensus")
    assert resp.status_code == 200
    assert resp.mimetype == "text/csv"
    assert "attachment; filename=atmosphere_anomalies_export.csv" in resp.headers["Content-Disposition"]

    content = resp.get_data(as_text=True)
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    # Header + 78 consensus anomalies
    assert len(rows) == 79
    assert rows[0][0] == "STATION_ID"
    assert rows[0][2] == "DATE"


def test_api_insights_endpoints(client):
    """Verify GET /api/insights/* return 200 with verified metrics."""
    endpoints = [
        ("/api/insights/summary", "baseline_metrics"),
        ("/api/insights/station-bias", "stations"),
        ("/api/insights/seasonal-bias", "seasons"),
        ("/api/insights/annual-bias", None),
        ("/api/insights/limitations", None),
        ("/api/insights/ethical-guidelines", None),
        ("/api/insights/methodology", "steps")
    ]
    for url, key in endpoints:
        resp = client.get(url)
        assert resp.status_code == 200, f"Failed for {url}"
        assert resp.json["success"] is True
        if key:
            assert key in resp.json["data"]


# =============================================================================
# 6. HTML VIEW RENDERING & REGRESSION TESTS
# =============================================================================

def test_anomaly_investigation_view_renders(client):
    """Verify GET /anomaly-investigation renders full Module 9 page."""
    resp = client.get("/anomaly-investigation")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Core titles and indicators
    assert "Anomaly Investigation &amp; Extreme Events" in html or "Anomaly Investigation & Extreme Events" in html
    assert "MODULE 09" in html
    assert "SIH26073" in html
    assert "Polar Vortex" in html
    assert "Hurricane Ida" in html
    assert "Heat Dome" in html
    assert "Severity Index" in html
    assert "chart-investigation-timeline" in html
    assert "observation-inspector-card" in html


def test_scientific_insights_view_renders(client):
    """Verify GET /scientific-insights renders full Module 10 page."""
    resp = client.get("/scientific-insights")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Core titles and insights
    assert "Scientific Insights" in html
    assert "MODULE 10" in html
    assert "The Jaccard Paradox" in html
    assert "12.50%" in html or "12.5%" in html
    assert "Geographical Station Bias" in html
    assert "Seasonal Variance Bias" in html
    assert "chart-station-bias" in html
    assert "chart-seasonal-bias" in html
    assert "Formal Catalog of Methodological Limitations" in html
    assert "Ethical &amp; Responsible" in html or "Ethical & Responsible" in html

