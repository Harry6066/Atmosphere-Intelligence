"""REST API Blueprint for Atmosphere Intelligence.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Serves verified experimental metrics, dataset summaries, model comparisons,
and health-check diagnostics in structured JSON.
"""

import logging
from flask import Blueprint, jsonify, request, current_app, Response

from app.services.data_service import DataService
from app.services.ml_service import MLService

logger = logging.getLogger("api")
api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health-check endpoint validating server status and dataset availability."""
    try:
        ds = DataService.get_instance()
        df = ds.get_raw_consensus_dataframe()
        return jsonify({
            "status": "healthy",
            "service": current_app.config.get("APP_NAME", "ATMOSPHERE INTELLIGENCE"),
            "problem_id": current_app.config.get("PROBLEM_ID", "SIH26073"),
            "course": current_app.config.get("COURSE", "INT396"),
            "dataset_loaded": True,
            "observations_in_memory": len(df),
            "stations_count": df["STATION"].nunique()
        }), 200
    except Exception as exc:
        logger.error("Health check failure: %s", exc)
        return jsonify({
            "status": "unhealthy",
            "error": str(exc),
            "dataset_loaded": False
        }), 500


@api_bp.route("/overview", methods=["GET"])
def get_overview():
    """Return top-level verified project metrics (observations, IF/LOF/consensus counts)."""
    try:
        ds = DataService.get_instance()
        metrics = ds.get_overview_metrics()
        return jsonify({
            "success": True,
            "data": metrics
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch overview metrics: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Internal error retrieving overview metrics: {str(exc)}"
        }), 500


@api_bp.route("/dataset/summary", methods=["GET"])
def get_dataset_summary():
    """Return weather station metadata, geographical coordinates, and feature definitions."""
    try:
        ds = DataService.get_instance()
        summary = ds.get_dataset_summary()
        return jsonify({
            "success": True,
            "data": summary
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch dataset summary: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Internal error retrieving dataset summary: {str(exc)}"
        }), 500


@api_bp.route("/comparison", methods=["GET"])
def get_comparison():
    """Return baseline IF vs LOF comparison and 9-cell cross-parameter sensitivity grid."""
    try:
        ds = DataService.get_instance()
        comp_data = ds.get_comparison_data()
        return jsonify({
            "success": True,
            "data": comp_data
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch comparison data: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Internal error retrieving model comparison: {str(exc)}"
        }), 500


@api_bp.route("/dataset/records", methods=["GET"])
@api_bp.route("/dataset/observations", methods=["GET"])
def get_dataset_records():
    """Return paginated, filtered weather observation records."""
    try:
        ds = DataService.get_instance()
        station = request.args.get("station", None)
        year = request.args.get("year", None)
        start_date = request.args.get("start_date", None)
        end_date = request.args.get("end_date", None)
        anomaly_type = request.args.get("anomaly_type", None)
        search = request.args.get("search", None)
        sort_col = request.args.get("sort_col", "date")
        sort_dir = request.args.get("sort_dir", "desc")
        
        # Validate sort direction
        if sort_dir.lower() not in ("asc", "desc"):
            sort_dir = "desc"

        # Whitelist allowed sort columns
        allowed_sorts = {
            "date", "station", "name", "tmax", "tmin", "temp_range", 
            "prcp", "snow", "snwd", "if_score", "lof_score", "consensus_label"
        }
        if sort_col.lower() not in allowed_sorts:
            sort_col = "date"

        try:
            page = int(request.args.get("page", 1))
        except (ValueError, TypeError):
            page = 1

        try:
            page_size = int(request.args.get("page_size", 25))
        except (ValueError, TypeError):
            page_size = 25

        result = ds.get_records(
            station=station,
            year=year,
            start_date=start_date,
            end_date=end_date,
            anomaly_type=anomaly_type,
            search=search,
            sort_col=sort_col,
            sort_dir=sort_dir,
            page=page,
            page_size=page_size
        )
        return jsonify({
            "success": True,
            "data": result["records"],
            "pagination": result["pagination"],
            "summary": result["summary"]
        }), 200
    except Exception as exc:
        logger.error("Failed to query dataset records: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error querying observation records: {str(exc)}"
        }), 500


@api_bp.route("/dataset/stations", methods=["GET"])
def get_dataset_stations():
    """Return weather station geographical metadata and coordinates."""
    try:
        ds = DataService.get_instance()
        summary = ds.get_dataset_summary()
        return jsonify({
            "success": True,
            "data": summary["stations"]
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch stations: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving stations: {str(exc)}"
        }), 500


@api_bp.route("/dataset/export", methods=["GET"])
def export_dataset_csv():
    """Export filtered observations subset as downloadable CSV file."""
    try:
        ds = DataService.get_instance()
        station = request.args.get("station", None)
        year = request.args.get("year", None)
        start_date = request.args.get("start_date", None)
        end_date = request.args.get("end_date", None)
        anomaly_type = request.args.get("anomaly_type", None)
        search = request.args.get("search", None)

        csv_content = ds.export_records_csv(
            station=station,
            year=year,
            start_date=start_date,
            end_date=end_date,
            anomaly_type=anomaly_type,
            search=search
        )
        filename = "noaa_weather_observations"
        if station and station.upper() != "ALL":
            filename += f"_{station.upper()}"
        if year and str(year).upper() != "ALL":
            filename += f"_{year}"
        filename += ".csv"

        return Response(
            csv_content,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as exc:
        logger.error("Failed to export dataset CSV: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error exporting CSV: {str(exc)}"
        }), 500


@api_bp.route("/dataset/quality-stats", methods=["GET"])
def get_dataset_quality_stats():
    """Return dataset missingness statistics and QC flag breakdown."""
    try:
        ds = DataService.get_instance()
        stats = ds.get_quality_and_flag_stats()
        return jsonify({
            "success": True,
            "data": stats
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch dataset quality stats: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving quality stats: {str(exc)}"
        }), 500


# -----------------------------------------------------------------------------
# EDA Endpoints (Module 3)
# -----------------------------------------------------------------------------

@api_bp.route("/eda/distributions", methods=["GET"])
def get_eda_distributions():
    """Return histogram, boxplot, and summary stats for a selected feature."""
    try:
        ds = DataService.get_instance()
        station = request.args.get("station", None)
        feature = request.args.get("feature", "TMAX")
        try:
            bins = int(request.args.get("bins", 30))
        except (ValueError, TypeError):
            bins = 30

        result = ds.get_eda_distributions(station=station, feature=feature, bins_count=bins)
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch EDA distributions: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error calculating distribution stats: {str(exc)}"
        }), 500


@api_bp.route("/eda/correlations", methods=["GET"])
def get_eda_correlations():
    """Return Pearson correlation matrix for numerical weather features."""
    try:
        ds = DataService.get_instance()
        station = request.args.get("station", None)
        result = ds.get_eda_correlations(station=station)
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch correlations: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error calculating correlation matrix: {str(exc)}"
        }), 500


@api_bp.route("/eda/seasonal", methods=["GET"])
def get_eda_seasonal():
    """Return monthly climatology and seasonal aggregations."""
    try:
        ds = DataService.get_instance()
        station = request.args.get("station", None)
        variable = request.args.get("variable", "TMAX")
        result = ds.get_eda_seasonal(station=station, variable=variable)
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch seasonal patterns: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error calculating seasonal climatology: {str(exc)}"
        }), 500


@api_bp.route("/eda/station-comparison", methods=["GET"])
def get_eda_station_comparison():
    """Return comparative climatological profiles across all monitored weather stations."""
    try:
        ds = DataService.get_instance()
        result = ds.get_eda_station_comparison()
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch station comparison: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error calculating station comparison: {str(exc)}"
        }), 500


# -----------------------------------------------------------------------------
# Data Quality & Preprocessing Endpoints (Module 4)
# -----------------------------------------------------------------------------

@api_bp.route("/preprocessing/audit", methods=["GET"])
def get_preprocessing_audit():
    """Return raw vs processed dataset audit and physical consistency check results."""
    try:
        ds = DataService.get_instance()
        result = ds.get_preprocessing_audit_details()
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch preprocessing audit: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving preprocessing audit: {str(exc)}"
        }), 500


@api_bp.route("/preprocessing/features", methods=["GET"])
def get_preprocessing_features():
    """Return catalog of engineered climatological features with formulas and rationales."""
    try:
        ds = DataService.get_instance()
        result = ds.get_preprocessing_features_catalog()
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch preprocessing features: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving features catalog: {str(exc)}"
        }), 500


@api_bp.route("/preprocessing/scaling", methods=["GET"])
def get_preprocessing_scaling():
    """Return before and after StandardScaler statistics table."""
    try:
        ds = DataService.get_instance()
        result = ds.get_scaling_comparison()
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch scaling comparison: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving scaling comparison: {str(exc)}"
        }), 500


# -----------------------------------------------------------------------------
# Module 5: Isolation Forest Endpoints
# -----------------------------------------------------------------------------

@api_bp.route("/isolation-forest/results", methods=["GET"])
def get_isolation_forest_results():
    """Return complete Isolation Forest analytics, distribution, and top suspicious observations."""
    try:
        ds = DataService.get_instance()
        raw_contamination = request.args.get("contamination", "0.03")
        raw_estimators = request.args.get("n_estimators", "100")
        station_filter = request.args.get("station", "ALL")

        try:
            contamination = float(raw_contamination)
        except ValueError:
            return jsonify({"success": False, "error": f"Invalid contamination value: '{raw_contamination}'. Must be numeric."}), 400

        try:
            n_estimators = int(raw_estimators)
        except ValueError:
            return jsonify({"success": False, "error": f"Invalid n_estimators value: '{raw_estimators}'. Must be an integer."}), 400

        result = ds.get_isolation_forest_analysis(
            contamination=contamination,
            n_estimators=n_estimators,
            station_filter=station_filter
        )
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch Isolation Forest results: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving Isolation Forest results: {str(exc)}"
        }), 500


@api_bp.route("/isolation-forest/run", methods=["POST"])
def run_isolation_forest_model():
    """Execute dynamic Isolation Forest training and inference with custom parameters."""
    try:
        body = request.get_json(silent=True) or {}
        ml = MLService.get_instance()

        raw_c = body.get("contamination", 0.03)
        raw_n = body.get("n_estimators", 100)
        max_samples = body.get("max_samples", "auto")
        raw_seed = body.get("random_state", 42)
        features = body.get("features", None)

        try:
            contamination = float(raw_c)
        except (ValueError, TypeError):
            return jsonify({"success": False, "error": f"Invalid contamination '{raw_c}'. Must be a float."}), 400

        try:
            n_estimators = int(raw_n)
        except (ValueError, TypeError):
            return jsonify({"success": False, "error": f"Invalid n_estimators '{raw_n}'. Must be an integer."}), 400

        try:
            random_state = int(raw_seed)
        except (ValueError, TypeError):
            return jsonify({"success": False, "error": f"Invalid random_state '{raw_seed}'. Must be an integer."}), 400

        result = ml.run_isolation_forest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            features=features
        )
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except ValueError as val_err:
        return jsonify({"success": False, "error": str(val_err)}), 400
    except Exception as exc:
        logger.error("Failed to execute Isolation Forest model: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error executing Isolation Forest: {str(exc)}"
        }), 500


@api_bp.route("/isolation-forest/sensitivity", methods=["GET"])
def get_isolation_forest_sensitivity():
    """Return 3-level contamination sensitivity matrix (0.01, 0.03, 0.05)."""
    try:
        ds = DataService.get_instance()
        result = ds.get_isolation_forest_sensitivity()
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch Isolation Forest sensitivity: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving sensitivity matrix: {str(exc)}"
        }), 500


# -----------------------------------------------------------------------------
# Module 6: Local Outlier Factor Endpoints
# -----------------------------------------------------------------------------

@api_bp.route("/lof/results", methods=["GET"])
def get_lof_results():
    """Return complete Local Outlier Factor analytics, NOF scores, and top suspicious observations."""
    try:
        ds = DataService.get_instance()
        raw_k = request.args.get("n_neighbors", "20")
        raw_c = request.args.get("contamination", "0.03")
        station_filter = request.args.get("station", "ALL")

        try:
            n_neighbors = int(raw_k)
        except ValueError:
            return jsonify({"success": False, "error": f"Invalid n_neighbors '{raw_k}'. Must be an integer."}), 400

        try:
            contamination = float(raw_c)
        except ValueError:
            return jsonify({"success": False, "error": f"Invalid contamination '{raw_c}'. Must be numeric."}), 400

        result = ds.get_lof_analysis(
            n_neighbors=n_neighbors,
            contamination=contamination,
            station_filter=station_filter
        )
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch LOF results: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving LOF results: {str(exc)}"
        }), 500


@api_bp.route("/lof/run", methods=["POST"])
def run_lof_model():
    """Execute dynamic Local Outlier Factor batch inference with custom parameters."""
    try:
        body = request.get_json(silent=True) or {}
        ml = MLService.get_instance()

        raw_k = body.get("n_neighbors", 20)
        raw_c = body.get("contamination", 0.03)
        metric = body.get("metric", "minkowski")
        features = body.get("features", None)

        try:
            n_neighbors = int(raw_k)
        except (ValueError, TypeError):
            return jsonify({"success": False, "error": f"Invalid n_neighbors '{raw_k}'. Must be an integer."}), 400

        try:
            contamination = float(raw_c)
        except (ValueError, TypeError):
            return jsonify({"success": False, "error": f"Invalid contamination '{raw_c}'. Must be numeric."}), 400

        result = ml.run_local_outlier_factor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            metric=metric,
            features=features
        )
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except ValueError as val_err:
        return jsonify({"success": False, "error": str(val_err)}), 400
    except Exception as exc:
        logger.error("Failed to execute LOF model: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error executing LOF: {str(exc)}"
        }), 500


@api_bp.route("/lof/sensitivity", methods=["GET"])
def get_lof_sensitivity():
    """Return neighborhood parameter sensitivity matrix (k = 10, 20, 30)."""
    try:
        ds = DataService.get_instance()
        result = ds.get_lof_sensitivity()
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch LOF sensitivity: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving LOF sensitivity matrix: {str(exc)}"
        }), 500


# -----------------------------------------------------------------------------
# Observation Investigation Detail Endpoint
# -----------------------------------------------------------------------------

@api_bp.route("/observation/inspect", methods=["GET"])
def inspect_observation():
    """Return full multi-feature profile and meteorological context for a specific observation."""
    try:
        station = request.args.get("station")
        date = request.args.get("date")

        if not station or not date:
            return jsonify({"success": False, "error": "Query parameters 'station' and 'date' are required."}), 400

        ds = DataService.get_instance()
        detail = ds.get_observation_inspection(station=station, date=date)
        return jsonify({
            "success": True,
            "data": detail
        }), 200
    except KeyError as key_err:
        return jsonify({"success": False, "error": str(key_err)}), 404
    except Exception as exc:
        logger.error("Failed to inspect observation: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving observation profile: {str(exc)}"
        }), 500


# -----------------------------------------------------------------------------
# Phase G: Model Comparison Engine Endpoints
# -----------------------------------------------------------------------------

@api_bp.route("/comparison/summary", methods=["GET"])
def get_comparison_summary():
    """Return baseline IF vs LOF comparison metrics, overlap stats, and correlation."""
    try:
        ds = DataService.get_instance()
        summary = ds.get_model_comparison_summary()
        return jsonify({
            "success": True,
            "data": summary
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch model comparison summary: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving comparison summary: {str(exc)}"
        }), 500


@api_bp.route("/comparison/station", methods=["GET"])
def get_comparison_station():
    """Return station-by-station breakdown of IF, LOF, and Consensus anomalies."""
    try:
        ds = DataService.get_instance()
        breakdown = ds.get_comparison_station_breakdown()
        return jsonify({
            "success": True,
            "data": breakdown
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch comparison station breakdown: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving station breakdown: {str(exc)}"
        }), 500


@api_bp.route("/comparison/seasonal", methods=["GET"])
def get_comparison_seasonal():
    """Return seasonal breakdown of IF, LOF, and Consensus anomalies."""
    try:
        ds = DataService.get_instance()
        breakdown = ds.get_comparison_seasonal_breakdown()
        return jsonify({
            "success": True,
            "data": breakdown
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch comparison seasonal breakdown: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving seasonal breakdown: {str(exc)}"
        }), 500


@api_bp.route("/comparison/yearly", methods=["GET"])
def get_comparison_yearly():
    """Return annual trend breakdown of IF, LOF, and Consensus anomalies."""
    try:
        ds = DataService.get_instance()
        breakdown = ds.get_comparison_yearly_breakdown()
        return jsonify({
            "success": True,
            "data": breakdown
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch comparison yearly breakdown: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving yearly breakdown: {str(exc)}"
        }), 500


@api_bp.route("/comparison/scatter", methods=["GET"])
def get_comparison_scatter():
    """Return bivariate scatter sample of IF vs LOF anomaly scores."""
    try:
        n_points = int(request.args.get("n_points", 1200))
        ds = DataService.get_instance()
        sample = ds.get_comparison_scatter_sample(n_points=n_points)
        return jsonify({
            "success": True,
            "data": sample
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch comparison scatter sample: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving scatter sample: {str(exc)}"
        }), 500


@api_bp.route("/comparison/observations", methods=["GET"])
def get_comparison_observations():
    """Return paginated and filtered observation comparison records."""
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 25))
        station = request.args.get("station")
        year_val = request.args.get("year")
        year = int(year_val) if year_val and year_val.isdigit() else None
        season = request.args.get("season")
        status = request.args.get("status")
        sort_by = request.args.get("sort_by", "DATE")
        sort_order = request.args.get("sort_order", "desc")

        ds = DataService.get_instance()
        result = ds.get_comparison_observations(
            page=page,
            per_page=per_page,
            station=station,
            year=year,
            season=season,
            status=status,
            sort_by=sort_by,
            sort_order=sort_order
        )
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch comparison observations: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving comparison observations: {str(exc)}"
        }), 500


@api_bp.route("/comparison/sensitivity", methods=["GET"])
def get_comparison_sensitivity():
    """Return the 9-cell cross-parameter sensitivity grid."""
    try:
        ds = DataService.get_instance()
        grid = ds.get_comparison_data().get("parameter_grid", [])
        return jsonify({
            "success": True,
            "data": grid
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch comparison sensitivity grid: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving sensitivity grid: {str(exc)}"
        }), 500


# -----------------------------------------------------------------------------
# Phase G: PCA Dimensionality Reduction Endpoints
# -----------------------------------------------------------------------------

@api_bp.route("/pca/summary", methods=["GET"])
def get_pca_summary():
    """Return PCA explained variance ratios (2D, 3D, all 10) and top drivers."""
    try:
        ds = DataService.get_instance()
        summary = ds.get_pca_summary()
        return jsonify({
            "success": True,
            "data": summary
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch PCA summary: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving PCA summary: {str(exc)}"
        }), 500


@api_bp.route("/pca/scree", methods=["GET"])
def get_pca_scree():
    """Return individual and cumulative explained variance for all 10 components."""
    try:
        ds = DataService.get_instance()
        scree = ds.get_pca_scree()
        return jsonify({
            "success": True,
            "data": scree
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch PCA scree data: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving PCA scree data: {str(exc)}"
        }), 500


@api_bp.route("/pca/loadings", methods=["GET"])
def get_pca_loadings():
    """Return eigenvector loadings matrix for all 10 features across PC1, PC2, and PC3."""
    try:
        ds = DataService.get_instance()
        loadings = ds.get_pca_loadings()
        return jsonify({
            "success": True,
            "data": loadings
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch PCA loadings: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving PCA loadings: {str(exc)}"
        }), 500


@api_bp.route("/pca/coordinates", methods=["GET"])
def get_pca_coordinates():
    """Return 2D/3D PCA projected coordinates with metadata for visualization."""
    try:
        sample = request.args.get("sample", "true").lower() in ("true", "1", "yes")
        color_by = request.args.get("color_by", "consensus")
        station = request.args.get("station")
        season = request.args.get("season")

        ds = DataService.get_instance()
        coords = ds.get_pca_coordinates(
            sample=sample,
            color_by=color_by,
            station_filter=station,
            season_filter=season
        )
        return jsonify({
            "success": True,
            "data": coords
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch PCA coordinates: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving PCA coordinates: {str(exc)}"
        }), 500


# =============================================================================
# PHASE H: MODULE 9 - ANOMALY INVESTIGATION & EXTREME EVENTS APIS
# =============================================================================

@api_bp.route("/investigation/summary", methods=["GET"])
def get_investigation_summary():
    """Return top-level summary metrics for Anomaly Investigation workspace."""
    try:
        ds = DataService.get_instance()
        summary = ds.get_investigation_summary()
        return jsonify({
            "success": True,
            "data": summary
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch investigation summary: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving investigation summary: {str(exc)}"
        }), 500


@api_bp.route("/investigation/observations", methods=["GET"])
def get_investigation_observations():
    """Return paginated, multi-filtered investigation observations."""
    try:
        station = request.args.get("station")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        year = request.args.get("year")
        season = request.args.get("season")
        classification = request.args.get("classification")
        consensus_status = request.args.get("consensus_status")
        qflag = request.args.get("qflag")
        search = request.args.get("search")
        sort_by = request.args.get("sort_by", "date")
        sort_dir = request.args.get("sort_dir", "desc")

        try:
            page = int(request.args.get("page", 1))
        except ValueError:
            page = 1

        try:
            per_page = int(request.args.get("per_page", 20))
        except ValueError:
            per_page = 20

        ds = DataService.get_instance()
        result = ds.get_investigation_observations(
            station=station,
            start_date=start_date,
            end_date=end_date,
            year=year,
            season=season,
            classification=classification,
            consensus_status=consensus_status,
            qflag=qflag,
            search=search,
            page=page,
            per_page=per_page,
            sort_by=sort_by,
            sort_dir=sort_dir
        )
        return jsonify({
            "success": True,
            "data": result
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch investigation observations: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving investigation observations: {str(exc)}"
        }), 500


@api_bp.route("/investigation/observation/<station>/<date>", methods=["GET"])
@api_bp.route("/investigation/observation", methods=["GET"])
def get_observation_detail(station=None, date=None):
    """Return single-observation investigation package with context window and percentiles."""
    try:
        st = station or request.args.get("station")
        dt = date or request.args.get("date")

        if not st or not dt:
            return jsonify({
                "success": False,
                "error": "Missing required parameters: 'station' and 'date' must be provided."
            }), 400

        ds = DataService.get_instance()
        detail = ds.get_observation_detail(station=st, date=dt)
        if detail is None:
            return jsonify({
                "success": False,
                "error": f"Observation not found for station '{st}' on date '{dt}'."
            }), 404

        return jsonify({
            "success": True,
            "data": detail
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch observation detail: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving observation detail: {str(exc)}"
        }), 500


@api_bp.route("/investigation/timeline", methods=["GET"])
def get_investigation_timeline():
    """Return monthly aggregated anomaly timeline for trend analysis."""
    try:
        station = request.args.get("station")
        year = request.args.get("year")

        ds = DataService.get_instance()
        timeline = ds.get_investigation_timeline(station=station, year=year)
        return jsonify({
            "success": True,
            "data": timeline
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch investigation timeline: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving investigation timeline: {str(exc)}"
        }), 500


@api_bp.route("/investigation/extreme-events", methods=["GET"])
def get_historical_extreme_events():
    """Return curated audit of documented real-world historical extreme events."""
    try:
        ds = DataService.get_instance()
        events = ds.get_historical_extreme_events()
        return jsonify({
            "success": True,
            "data": events
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch historical extreme events: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving historical extreme events: {str(exc)}"
        }), 500


@api_bp.route("/investigation/top-suspicious", methods=["GET"])
def get_top_suspicious_observations():
    """Return top consensus anomalies ranked by severity index."""
    try:
        limit = int(request.args.get("limit", 20))
        ds = DataService.get_instance()
        records = ds.get_top_suspicious_observations(limit=limit)
        return jsonify({
            "success": True,
            "data": records
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch top suspicious observations: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving top suspicious observations: {str(exc)}"
        }), 500


@api_bp.route("/investigation/station-context", methods=["GET"])
def get_station_context():
    """Return station climatological percentiles across key weather variables."""
    try:
        station = request.args.get("station")
        if not station:
            return jsonify({
                "success": False,
                "error": "Query parameter 'station' is required."
            }), 400

        ds = DataService.get_instance()
        # Find any observation from this station to calculate percentiles
        obs_df = ds.get_raw_consensus_dataframe()
        st_df = obs_df[obs_df["STATION"] == station]
        if len(st_df) == 0:
            return jsonify({
                "success": False,
                "error": f"Station '{station}' not found."
            }), 404

        percentiles = {}
        for col in ["TMAX", "TMIN", "TEMP_RANGE", "PRCP", "SNOW", "SNWD"]:
            if col in st_df.columns:
                s = st_df[col]
                percentiles[col] = {
                    "min": round(float(s.min()), 1),
                    "p10": round(float(s.quantile(0.1)), 1),
                    "p25": round(float(s.quantile(0.25)), 1),
                    "p50": round(float(s.median()), 1),
                    "p75": round(float(s.quantile(0.75)), 1),
                    "p90": round(float(s.quantile(0.9)), 1),
                    "p99": round(float(s.quantile(0.99)), 1),
                    "max": round(float(s.max()), 1),
                    "mean": round(float(s.mean()), 1),
                    "std": round(float(s.std()), 1)
                }

        return jsonify({
            "success": True,
            "data": {
                "station_id": station,
                "records_count": len(st_df),
                "percentiles": percentiles
            }
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch station context: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving station context: {str(exc)}"
        }), 500


@api_bp.route("/investigation/export", methods=["GET"])
def export_investigation_anomalies():
    """Export filtered anomaly records as a downloadable CSV stream."""
    try:
        import io
        import csv

        station = request.args.get("station")
        year = request.args.get("year")
        season = request.args.get("season")
        classification = request.args.get("classification", "any_anomaly")
        consensus_status = request.args.get("consensus_status")
        qflag = request.args.get("qflag")
        search = request.args.get("search")

        ds = DataService.get_instance()
        # Fetch all matching records without pagination limit
        result = ds.get_investigation_observations(
            station=station,
            year=year,
            season=season,
            classification=classification,
            consensus_status=consensus_status,
            qflag=qflag,
            search=search,
            page=1,
            per_page=12000,
            sort_by="date",
            sort_dir="desc"
        )

        output = io.StringIO()
        writer = csv.writer(output)

        # CSV Header
        writer.writerow([
            "STATION_ID",
            "STATION_NAME",
            "DATE",
            "SEASON",
            "YEAR",
            "LATITUDE",
            "LONGITUDE",
            "ELEVATION_M",
            "TMAX_C",
            "TMIN_C",
            "TEMP_RANGE_C",
            "PRCP_MM",
            "SNOW_MM",
            "SNWD_MM",
            "TMAX_DIFF_1D_C",
            "IF_ANOMALY",
            "IF_SCORE",
            "LOF_ANOMALY",
            "LOF_SCORE",
            "CONSENSUS_LABEL",
            "SEVERITY_INDEX",
            "INVESTIGATION_STATUS",
            "QC_FLAGGED"
        ])

        for item in result["items"]:
            writer.writerow([
                item["station_id"],
                item["station_name"],
                item["date"],
                item["season"],
                item["year"],
                item["latitude"],
                item["longitude"],
                item["elevation_m"],
                item["tmax"],
                item["tmin"],
                item["temp_range"],
                item["prcp"],
                item["snow"],
                item["snwd"],
                item["tmax_diff_1d"],
                item["if_anomaly"],
                item["if_score"],
                item["lof_anomaly"],
                item["lof_score"],
                item["consensus_label"],
                item["severity_index"],
                item["investigation_status"],
                item["qc_flagged"]
            ])

        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=atmosphere_anomalies_export.csv"}
        )
    except Exception as exc:
        logger.error("Failed to export anomalies CSV: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error generating anomaly CSV export: {str(exc)}"
        }), 500


# =============================================================================
# PHASE H: MODULE 10 - SCIENTIFIC INSIGHTS & LIMITATIONS APIS
# =============================================================================

@api_bp.route("/insights/summary", methods=["GET"])
def get_insights_summary():
    """Return executive scientific summary with baseline metrics and conclusions."""
    try:
        ds = DataService.get_instance()
        summary = ds.get_scientific_summary()
        return jsonify({
            "success": True,
            "data": summary
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch insights summary: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving insights summary: {str(exc)}"
        }), 500


@api_bp.route("/insights/station-bias", methods=["GET"])
def get_insights_station_bias():
    """Return geographical station bias analysis with explicit denominators."""
    try:
        ds = DataService.get_instance()
        bias = ds.get_station_bias_insights()
        return jsonify({
            "success": True,
            "data": bias
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch station bias insights: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving station bias insights: {str(exc)}"
        }), 500


@api_bp.route("/insights/seasonal-bias", methods=["GET"])
def get_insights_seasonal_bias():
    """Return seasonal variance bias analysis with explicit denominators."""
    try:
        ds = DataService.get_instance()
        bias = ds.get_seasonal_bias_insights()
        return jsonify({
            "success": True,
            "data": bias
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch seasonal bias insights: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving seasonal bias insights: {str(exc)}"
        }), 500


@api_bp.route("/insights/annual-bias", methods=["GET"])
def get_insights_annual_bias():
    """Return annual anomaly rates (2016-2023) with explicit denominators."""
    try:
        ds = DataService.get_instance()
        annual = ds.get_annual_bias_insights()
        return jsonify({
            "success": True,
            "data": annual
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch annual bias insights: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving annual bias insights: {str(exc)}"
        }), 500


@api_bp.route("/insights/limitations", methods=["GET"])
def get_insights_limitations():
    """Return formal catalog of all 9 methodological limitations in the study."""
    try:
        ds = DataService.get_instance()
        limitations = ds.get_scientific_limitations()
        return jsonify({
            "success": True,
            "data": limitations
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch scientific limitations: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving scientific limitations: {str(exc)}"
        }), 500


@api_bp.route("/insights/ethical-guidelines", methods=["GET"])
def get_insights_ethical_guidelines():
    """Return 6 core ethical and responsible AI guidelines for meteorological quality control."""
    try:
        ds = DataService.get_instance()
        guidelines = ds.get_ethical_guidelines()
        return jsonify({
            "success": True,
            "data": guidelines
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch ethical guidelines: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving ethical guidelines: {str(exc)}"
        }), 500


@api_bp.route("/insights/methodology", methods=["GET"])
def get_insights_methodology():
    """Return complete 10-step reproducibility methodology workflow."""
    try:
        ds = DataService.get_instance()
        workflow = ds.get_methodology_workflow()
        return jsonify({
            "success": True,
            "data": workflow
        }), 200
    except Exception as exc:
        logger.error("Failed to fetch methodology workflow: %s", exc)
        return jsonify({
            "success": False,
            "error": f"Error retrieving methodology workflow: {str(exc)}"
        }), 500





