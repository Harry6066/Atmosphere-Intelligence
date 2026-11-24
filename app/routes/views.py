"""HTML View Blueprint for Atmosphere Intelligence.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Renders master Jinja2 templates, active navigation states, and planned module shells.
"""

from flask import Blueprint, render_template, current_app
from app.services.data_service import DataService
from src.data_preprocessing import NUMERICAL_FEATURE_COLS

views_bp = Blueprint("views", __name__)


@views_bp.route("/")
def index():
    """Module 1: Project Overview (Landing Dashboard)."""
    ds = DataService.get_instance()
    overview_metrics = ds.get_overview_metrics()
    return render_template(
        "index.html",
        active_page="overview",
        module_category="Executive",
        module_title="Project Overview",
        overview=overview_metrics
    )


@views_bp.route("/dataset-explorer")
def dataset_explorer():
    """Module 2: Dataset Explorer (Phase D Complete)."""
    ds = DataService.get_instance()
    summary = ds.get_dataset_summary()
    quality = ds.get_quality_and_flag_stats()
    return render_template(
        "dataset_explorer.html",
        active_page="dataset",
        module_category="Data Foundation",
        module_title="Dataset Explorer",
        stations=summary["stations"],
        feature_definitions=summary["feature_definitions"],
        total_records=summary["total_records"],
        quality=quality
    )


@views_bp.route("/eda")
def eda():
    """Module 3: Exploratory Data Analysis (Phase E Active)."""
    ds = DataService.get_instance()
    summary = ds.get_dataset_summary()
    station_comp = ds.get_eda_station_comparison()
    init_dist = ds.get_eda_distributions(feature="TMAX")
    init_corr = ds.get_eda_correlations()
    init_seasonal = ds.get_eda_seasonal(variable="TMAX")
    return render_template(
        "eda.html",
        active_page="eda",
        module_category="Data Foundation",
        module_title="Exploratory Data Analysis",
        stations=summary["stations"],
        station_comparison=station_comp["stations"],
        initial_distribution=init_dist,
        initial_correlation=init_corr,
        initial_seasonal=init_seasonal,
        total_records=summary["total_records"]
    )


@views_bp.route("/data-quality")
def data_quality():
    """Module 4: Data Quality & Preprocessing (Phase E Active)."""
    ds = DataService.get_instance()
    summary = ds.get_dataset_summary()
    audit_details = ds.get_preprocessing_audit_details()
    features_catalog = ds.get_preprocessing_features_catalog()
    scaling_comp = ds.get_scaling_comparison()
    return render_template(
        "data_quality.html",
        active_page="preprocessing",
        module_category="Data Foundation",
        module_title="Data Quality & Preprocessing",
        stations=summary["stations"],
        audit=audit_details,
        features=features_catalog["features"],
        scaling=scaling_comp,
        total_records=summary["total_records"]
    )


@views_bp.route("/isolation-forest")
def isolation_forest():
    """Module 5: Isolation Forest Anomaly Detection (Phase F Complete)."""
    ds = DataService.get_instance()
    summary = ds.get_dataset_summary()
    initial_analysis = ds.get_isolation_forest_analysis(contamination=0.03, n_estimators=100)
    sensitivity = ds.get_isolation_forest_sensitivity()
    return render_template(
        "isolation_forest.html",
        active_page="isolation_forest",
        module_category="Anomaly Models",
        module_title="Isolation Forest Detection",
        stations=summary["stations"],
        feature_list=NUMERICAL_FEATURE_COLS,
        initial_analysis=initial_analysis,
        sensitivity=sensitivity,
        total_records=summary["total_records"]
    )


@views_bp.route("/local-outlier-factor")
@views_bp.route("/lof")
def local_outlier_factor():
    """Module 6: Local Outlier Factor (LOF) Anomaly Detection (Phase F Complete)."""
    ds = DataService.get_instance()
    summary = ds.get_dataset_summary()
    initial_analysis = ds.get_lof_analysis(n_neighbors=20, contamination=0.03)
    sensitivity = ds.get_lof_sensitivity()
    return render_template(
        "lof.html",
        active_page="lof",
        module_category="Anomaly Models",
        module_title="Local Outlier Factor (LOF)",
        stations=summary["stations"],
        feature_list=NUMERICAL_FEATURE_COLS,
        initial_analysis=initial_analysis,
        sensitivity=sensitivity,
        total_records=summary["total_records"]
    )


@views_bp.route("/model-comparison")
def model_comparison():
    """Module 7: Model Comparison Engine (Phase G Complete)."""
    ds = DataService.get_instance()
    summary = ds.get_dataset_summary()
    comp_summary = ds.get_model_comparison_summary()
    station_breakdown = ds.get_comparison_station_breakdown()
    seasonal_breakdown = ds.get_comparison_seasonal_breakdown()
    sensitivity_grid = ds.get_comparison_data().get("parameter_grid", [])
    initial_obs = ds.get_comparison_observations(page=1, per_page=15)
    return render_template(
        "model_comparison.html",
        active_page="comparison",
        module_category="Evaluation & Projections",
        module_title="Model Comparison Engine",
        stations=summary["stations"],
        comp_summary=comp_summary,
        station_breakdown=station_breakdown,
        seasonal_breakdown=seasonal_breakdown,
        sensitivity_grid=sensitivity_grid,
        initial_observations=initial_obs,
        total_records=summary["total_records"]
    )


@views_bp.route("/pca-visualization")
@views_bp.route("/pca-analysis")
def pca_visualization():
    """Module 8: PCA 2D & 3D Dimensionality Reduction (Phase G Complete)."""
    ds = DataService.get_instance()
    summary = ds.get_dataset_summary()
    pca_summary = ds.get_pca_summary()
    scree_data = ds.get_pca_scree()
    loadings_data = ds.get_pca_loadings()
    return render_template(
        "pca_analysis.html",
        active_page="pca",
        module_category="Evaluation & Projections",
        module_title="PCA Dimensionality Reduction",
        stations=summary["stations"],
        pca_summary=pca_summary,
        scree_data=scree_data,
        loadings_data=loadings_data,
        total_records=summary["total_records"]
    )


@views_bp.route("/anomaly-investigation")
def anomaly_investigation():
    """Module 9: Anomaly Investigation & Extreme Events (Phase H Complete)."""
    ds = DataService.get_instance()
    summary = ds.get_investigation_summary()
    stations = ds.get_dataset_summary()["stations"]
    extreme_events = ds.get_historical_extreme_events()
    top_suspicious = ds.get_top_suspicious_observations(limit=10)
    initial_obs = ds.get_investigation_observations(page=1, per_page=15)
    return render_template(
        "anomaly_investigation.html",
        active_page="investigation",
        module_category="Scientific Validation",
        module_title="Anomaly Investigation & Events",
        summary=summary,
        stations=stations,
        extreme_events=extreme_events,
        top_suspicious=top_suspicious,
        initial_observations=initial_obs,
        total_records=summary["total_observations"]
    )


@views_bp.route("/scientific-insights")
def scientific_insights():
    """Module 10: Scientific Insights & Limitations (Phase H Complete)."""
    ds = DataService.get_instance()
    summary = ds.get_scientific_summary()
    station_bias = ds.get_station_bias_insights()
    seasonal_bias = ds.get_seasonal_bias_insights()
    annual_bias = ds.get_annual_bias_insights()
    limitations = ds.get_scientific_limitations()
    ethical_guidelines = ds.get_ethical_guidelines()
    methodology = ds.get_methodology_workflow()
    stations = ds.get_dataset_summary()["stations"]
    return render_template(
        "scientific_insights.html",
        active_page="insights",
        module_category="Scientific Validation",
        module_title="Scientific Insights & Limitations",
        summary=summary,
        station_bias=station_bias,
        seasonal_bias=seasonal_bias,
        annual_bias=annual_bias,
        limitations=limitations,
        ethical_guidelines=ethical_guidelines,
        methodology=methodology,
        stations=stations,
        total_records=summary["baseline_metrics"]["total_observations"]
    )
