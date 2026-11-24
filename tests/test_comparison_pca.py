"""Unit tests for Atmosphere Intelligence Phase G: Model Comparison & PCA Projections.

Verifies:
- Module 7: Model Comparison Engine views, KPI summaries, station/season breakdowns,
  decision quadrant scatter, 9-cell parameter sensitivity grid, and filtered observations.
- Module 8: PCA 2D/3D Dimensionality Reduction views, variance ratios, scree spectrum,
  eigenvector loadings matrix, and projected coordinate datasets.
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
    """Create Flask application in testing mode."""
    app = create_app("testing")
    yield app


@pytest.fixture(scope="module")
def client(app):
    """Create Flask test client."""
    return app.test_client()


# =============================================================================
# MODULE 7: MODEL COMPARISON ENGINE TESTS
# =============================================================================

def test_model_comparison_view_renders_cleanly(client):
    """Verify GET /model-comparison renders master layout with real comparison KPIs."""
    resp = client.get("/model-comparison")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    assert "Model Comparison & Consensus Engine" in html
    assert "11,688" in html or "11688" in html
    assert "351" in html
    assert "78" in html
    assert "95.33%" in html or "95.33" in html
    assert "0.125" in html
    assert "Polar Vortex" in html
    assert "chart-venn-donut" in html
    assert "chart-score-scatter" in html


def test_model_comparison_summary_api(client):
    """Verify GET /api/comparison/summary delivers exact verified scientific metrics."""
    resp = client.get("/api/comparison/summary")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True

    data = payload["data"]
    assert data["total_observations"] == 11688
    assert data["if_anomalies"] == 351
    assert data["lof_anomalies"] == 351
    assert data["consensus_anomalies"] == 78
    assert data["if_only_anomalies"] == 273
    assert data["lof_only_anomalies"] == 273
    assert data["normal_both"] == 11064
    assert data["total_union_anomalies"] == 624
    assert data["binary_agreement_pct"] == 95.33
    assert data["jaccard_similarity"] == 0.125
    assert data["score_correlation"] == 0.4807


def test_comparison_station_breakdown_api(client):
    """Verify GET /api/comparison/station returns station-by-station anomaly counts."""
    resp = client.get("/api/comparison/station")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True

    items = payload["data"]
    assert len(items) == 4

    station_dict = {item["station_id"]: item for item in items}
    
    # Chicago (USW00094846): Polar vortex hotspot
    chi = station_dict["USW00094846"]
    assert chi["if_anomalies"] == 227
    assert chi["lof_anomalies"] == 177
    assert chi["consensus_anomalies"] == 58

    # Miami (USW00012839): Dense tropical cluster (LOF detects local shifts, IF rarely triggers)
    mia = station_dict["USW00012839"]
    assert mia["if_anomalies"] == 7
    assert mia["lof_anomalies"] == 87
    assert mia["consensus_anomalies"] == 2

    # New York (USW00094728)
    jfk = station_dict["USW00094728"]
    assert jfk["if_anomalies"] == 89
    assert jfk["lof_anomalies"] == 54
    assert jfk["consensus_anomalies"] == 16

    # Seattle (USW00024233)
    sea = station_dict["USW00024233"]
    assert sea["if_anomalies"] == 28
    assert sea["lof_anomalies"] == 33
    assert sea["consensus_anomalies"] == 2


def test_comparison_seasonal_breakdown_api(client):
    """Verify GET /api/comparison/seasonal returns 4 seasonal quarters."""
    resp = client.get("/api/comparison/seasonal")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True

    items = payload["data"]
    assert len(items) == 4
    seasons = {item["season"]: item for item in items}

    # Winter has maximum IF anomalies due to extreme sub-zero swings
    assert seasons["Winter"]["if_anomalies"] == 279
    assert seasons["Winter"]["lof_anomalies"] == 154
    assert seasons["Winter"]["consensus_anomalies"] == 57

    # Summer has fewest anomalies
    assert seasons["Summer"]["if_anomalies"] == 5
    assert seasons["Summer"]["lof_anomalies"] == 39
    assert seasons["Summer"]["consensus_anomalies"] == 3


def test_comparison_yearly_breakdown_api(client):
    """Verify GET /api/comparison/yearly returns annual trajectory across 8 years (2016-2023)."""
    resp = client.get("/api/comparison/yearly")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True

    items = payload["data"]
    assert len(items) == 8
    years = [item["year"] for item in items]
    assert years == list(range(2016, 2024))
    assert sum(item["if_anomalies"] for item in items) == 351
    assert sum(item["lof_anomalies"] for item in items) == 351
    assert sum(item["consensus_anomalies"] for item in items) == 78


def test_comparison_scatter_api(client):
    """Verify GET /api/comparison/scatter returns bivariate decision coordinates."""
    resp = client.get("/api/comparison/scatter?n_points=600")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True

    points = payload["data"]
    assert len(points) >= 624  # All union anomalies are guaranteed preserved

    # Check structure
    pt = points[0]
    assert "station_id" in pt
    assert "date" in pt
    assert "if_score" in pt
    assert "lof_score" in pt
    assert "consensus" in pt

    # Verify all 78 consensus anomalies are represented
    consensus_pts = [p for p in points if p["consensus"] == "Consensus"]
    assert len(consensus_pts) == 78


def test_comparison_observations_pagination_and_filters(client):
    """Verify GET /api/comparison/observations filters and paginates correctly."""
    # Filter by Consensus only
    resp = client.get("/api/comparison/observations?status=Consensus&page=1&per_page=20")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["total"] == 78
    assert len(data["records"]) == 20
    assert all(r["consensus"] == "Consensus" for r in data["records"])

    # Filter by Station Chicago and Winter
    resp2 = client.get("/api/comparison/observations?station=USW00094846&season=Winter&page=1&per_page=10")
    assert resp2.status_code == 200
    data2 = resp2.get_json()["data"]
    assert all(r["station_id"] == "USW00094846" for r in data2["records"])
    assert all(r["season"] == "Winter" for r in data2["records"])


def test_comparison_sensitivity_grid_api(client):
    """Verify GET /api/comparison/sensitivity delivers the 9-cell cross-parameter matrix."""
    resp = client.get("/api/comparison/sensitivity")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True

    grid = payload["data"]
    assert len(grid) == 9

    # Baseline cell (c=0.03, k=20)
    baseline_cell = next(
        c for c in grid if c["contamination"] == 0.03 and c["n_neighbors"] == 20
    )
    assert baseline_cell["consensus_anomalies"] == 78
    assert baseline_cell["jaccard_similarity"] == 0.125
    assert baseline_cell["if_anomalies"] == 351
    assert baseline_cell["lof_anomalies"] == 351


# =============================================================================
# MODULE 8: PCA DIMENSIONALITY REDUCTION TESTS
# =============================================================================

def test_pca_view_renders_cleanly(client):
    """Verify GET /pca-visualization and /pca-analysis render master layout with PCA KPIs."""
    for path in ["/pca-visualization", "/pca-analysis"]:
        resp = client.get(path)
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)

        assert "PCA 2D & 3D Dimensionality Reduction" in html
        assert "27.96%" in html
        assert "16.08%" in html
        assert "12.20%" in html
        assert "44.05%" in html
        assert "56.25%" in html
        assert "chart-pca-2d" in html
        assert "chart-pca-scree" in html
        assert "chart-pca-loadings" in html


def test_pca_summary_api(client):
    """Verify GET /api/pca/summary returns accurate explained variance metrics."""
    resp = client.get("/api/pca/summary")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True

    data = payload["data"]
    assert data["n_observations"] == 11688
    assert data["n_components"] == 10
    assert data["pc1_variance_pct"] == 27.96
    assert data["pc2_variance_pct"] == 16.08
    assert data["pc3_variance_pct"] == 12.20
    assert data["variance_2d_pct"] == 44.05
    assert data["variance_3d_pct"] == 56.25

    # Top drivers
    assert "TMAX_SCALED" in data["top_drivers"]["pc1"]
    assert "TMAX_DIFF_1D_SCALED" in data["top_drivers"]["pc2"]
    assert "TMIN_DIFF_1D_SCALED" in data["top_drivers"]["pc3"]


def test_pca_scree_api(client):
    """Verify GET /api/pca/scree returns all 10 components with rank-9 degeneracy."""
    resp = client.get("/api/pca/scree")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True

    scree = payload["data"]["components"]
    assert len(scree) == 10

    # PC1 variance
    assert scree[0]["component"] == 1
    assert scree[0]["explained_variance_pct"] == 27.96

    # PC10 variance is 0.0% due to TEMP_RANGE = TMAX - TMIN collinearity
    assert scree[9]["component"] == 10
    assert scree[9]["explained_variance_pct"] == 0.0
    assert scree[9]["cumulative_variance_pct"] == 100.0


def test_pca_loadings_api(client):
    """Verify GET /api/pca/loadings returns 10 feature loadings across PC1, PC2, and PC3."""
    resp = client.get("/api/pca/loadings")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True

    loadings = payload["data"]["loadings"]
    assert len(loadings) == 10

    fdict = {item["feature"]: item for item in loadings}
    
    # PC1: Thermal (TMAX: +0.5715, TMIN: +0.5333)
    assert fdict["TMAX"]["pc1"] == 0.5715
    assert fdict["TMIN"]["pc1"] == 0.5333

    # PC2: Frontal volatility (TMAX_DIFF_1D: +0.6624)
    assert fdict["TMAX_DIFF_1D"]["pc2"] == 0.6624

    # PC3: Hydrometeorology (PRCP: +0.4652, TMIN_DIFF: +0.5265, TEMP_RANGE: -0.4540)
    assert fdict["PRCP"]["pc3"] == 0.4652
    assert fdict["TMIN_DIFF_1D"]["pc3"] == 0.5265
    assert fdict["TEMP_RANGE"]["pc3"] == -0.4540


def test_pca_coordinates_api(client):
    """Verify GET /api/pca/coordinates delivers projected coordinates."""
    resp = client.get("/api/pca/coordinates?sample=true&color_by=consensus")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["success"] is True

    pts = payload["data"]
    assert len(pts) >= 1600  # 624 anomalies + 1000 sampled normals

    p0 = pts[0]
    assert "pc1" in p0
    assert "pc2" in p0
    assert "pc3" in p0
    assert "station_id" in p0
    assert "station" in p0
    assert "date" in p0
    assert "consensus" in p0

    # Test station filter
    resp_chi = client.get("/api/pca/coordinates?station=USW00094846")
    assert resp_chi.status_code == 200
    chi_pts = resp_chi.get_json()["data"]
    assert all(p["station_id"] == "USW00094846" for p in chi_pts)
