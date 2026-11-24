"""Unit tests for Atmosphere Intelligence master layout, views, and static assets."""

import pytest
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app import create_app


@pytest.fixture(scope="module")
def app():
    """Create Flask application in testing mode."""
    app = create_app("testing")
    yield app


@pytest.fixture(scope="module")
def client(app):
    """Create test client."""
    return app.test_client()


def test_landing_page_renders_with_real_metrics(client):
    """Verify GET / renders master layout with actual project metrics."""
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    
    # Assert core branding and problem statement
    assert "ATMOSPHERE" in html
    assert "SIH26073" in html
    assert "INT396" in html
    
    # Assert real metrics present in HTML
    assert "11,688" in html or "11688" in html
    assert "351" in html
    assert "78" in html
    assert "44.05%" in html or "44.05" in html


def test_all_ten_module_routes_exist(client):
    """Verify that all 10 module destinations return 200 OK with no 404s."""
    routes = [
        ("/", "Overview"),
        ("/dataset-explorer", "Dataset Explorer"),
        ("/eda", "Exploratory Data Analysis"),
        ("/data-quality", "Data Quality"),
        ("/isolation-forest", "Isolation Forest"),
        ("/local-outlier-factor", "Local Outlier Factor"),
        ("/model-comparison", "Model Comparison"),
        ("/pca-visualization", "PCA"),
        ("/anomaly-investigation", "Anomaly Investigation"),
        ("/scientific-insights", "Scientific Insights"),
    ]
    
    for route, expected_text in routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"Route {route} failed with {resp.status_code}"
        html = resp.get_data(as_text=True)
        assert expected_text in html, f"Expected '{expected_text}' in route {route}"


def test_static_css_and_js_served(client):
    """Verify atmosphere.css and core.js are served cleanly."""
    css_resp = client.get("/static/css/atmosphere.css")
    assert css_resp.status_code == 200
    assert "--bg-app" in css_resp.get_data(as_text=True)
    assert "--accent-cyan" in css_resp.get_data(as_text=True)
    
    js_resp = client.get("/static/js/core.js")
    assert js_resp.status_code == 200
    assert "Atmosphere" in js_resp.get_data(as_text=True)
    assert "initTheme" in js_resp.get_data(as_text=True)


def test_theme_anti_flash_script(client):
    """Verify that base template contains anti-flash script and data-theme default."""
    resp = client.get("/")
    html = resp.get_data(as_text=True)
    assert 'data-theme="dark"' in html
    assert "atmosphere-theme" in html
