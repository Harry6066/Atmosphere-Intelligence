"""MLService: Unsupervised Machine Learning Model Wrapper.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Integrates with existing Isolation Forest, Local Outlier Factor, PCA, and
comparison modules in src/. Caches baseline runs and executes validated
dynamic parameter experiments with empirical runtime measurements.
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data_preprocessing import NUMERICAL_FEATURE_COLS
from src.isolation_forest import train_and_predict_isolation_forest, RANDOM_STATE
from src.lof_detection import train_and_predict_lof
from src.comparison import compute_anomaly_metrics
from app.services.data_service import DataService

logger = logging.getLogger("ml_service")


class MLService:
    """Service wrapping Isolation Forest, LOF, and PCA execution."""
    
    _instance: Optional["MLService"] = None
    
    def __init__(self, data_service: Optional[DataService] = None):
        if data_service is None:
            data_service = DataService.get_instance()
        self.data_service = data_service
        self._scaled_features = [f"{c}_SCALED" for c in NUMERICAL_FEATURE_COLS]
        self._if_dynamic_cache: Dict[str, Dict[str, Any]] = {}
        self._lof_dynamic_cache: Dict[str, Dict[str, Any]] = {}
        
    @classmethod
    def get_instance(cls, data_service: Optional[DataService] = None) -> "MLService":
        if cls._instance is None:
            cls._instance = cls(data_service)
        return cls._instance

    def run_isolation_forest(
        self,
        contamination: float = 0.03,
        n_estimators: int = 100,
        max_samples: Union[str, int] = "auto",
        random_state: int = RANDOM_STATE,
        features: Optional[list] = None
    ) -> Dict[str, Any]:
        """Execute Isolation Forest with input validation, caching, and runtime benchmarking.
        
        If contamination is in precomputed set [0.01, 0.03, 0.05] with default parameters,
        returns verified baseline metrics instantly. Otherwise, runs live Isolation Forest.
        """
        # Parameter validation
        if not (0.005 <= contamination <= 0.20):
            raise ValueError(f"Contamination must be between 0.005 and 0.20; got {contamination}")
        if not (10 <= n_estimators <= 300):
            raise ValueError(f"n_estimators must be between 10 and 300; got {n_estimators}")
        if max_samples != "auto":
            try:
                max_samples = int(max_samples)
                if not (32 <= max_samples <= 5000):
                    raise ValueError("max_samples must be between 32 and 5000")
            except (TypeError, ValueError):
                raise ValueError(f"Invalid max_samples parameter: {max_samples}")

        # Feature selection
        if features is None or len(features) == 0:
            selected_features = self._scaled_features
        else:
            selected_features = []
            for f in features:
                clean_f = f.replace("_SCALED", "")
                if clean_f not in NUMERICAL_FEATURE_COLS:
                    raise ValueError(f"Invalid feature '{f}'. Valid features: {NUMERICAL_FEATURE_COLS}")
                selected_features.append(f"{clean_f}_SCALED")

        # Cache key for custom runs
        cache_key = f"IF_{round(contamination, 3)}_{n_estimators}_{max_samples}_{random_state}_{','.join(sorted(selected_features))}"
        if cache_key in self._if_dynamic_cache:
            cached_result = dict(self._if_dynamic_cache[cache_key])
            cached_result["execution"] = dict(cached_result["execution"])
            cached_result["execution"]["cached"] = True
            return cached_result

        df = self.data_service.get_raw_consensus_dataframe()
        c_round = round(contamination, 2)
        precomputed_cols = {
            0.01: ("IF_ANOMALY_C01", "IF_SCORE_C01"),
            0.03: ("IF_ANOMALY", "IF_SCORE"),
            0.05: ("IF_ANOMALY_C05", "IF_SCORE_C05")
        }

        t0 = time.perf_counter()
        is_baseline = (
            c_round in precomputed_cols
            and n_estimators == 100
            and max_samples == "auto"
            and random_state == RANDOM_STATE
            and selected_features == self._scaled_features
        )

        if is_baseline:
            anom_col, score_col = precomputed_cols[c_round]
            labels = df[anom_col].values
            scores = df[score_col].values
            was_cached = True
        else:
            X_scaled = df[selected_features].values
            labels, scores, _ = train_and_predict_isolation_forest(
                X_scaled,
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=random_state
            )
            was_cached = False

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)

        n_anomalies = int(labels.sum())
        n_normal = len(labels) - n_anomalies
        pct_anomalies = round((n_anomalies / len(labels)) * 100, 2)

        # Histogram: 30 bins
        hist_counts, bin_edges = np.histogram(scores, bins=30)
        bin_centers = [round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 4) for i in range(len(hist_counts))]
        anom_mask = (labels == 1)
        norm_mask = (labels == 0)
        hist_anom, _ = np.histogram(scores[anom_mask], bins=bin_edges)
        hist_norm, _ = np.histogram(scores[norm_mask], bins=bin_edges)

        # Station distribution
        st_names = {
            "USW00094846": "Chicago O'Hare, IL",
            "USW00094728": "NY Central Park, NY",
            "USW00024233": "Seattle Tacoma, WA",
            "USW00012839": "Miami Intl Airport, FL"
        }
        station_dist = []
        for st_id, st_name in st_names.items():
            mask = (df["STATION"] == st_id)
            st_anom = int(labels[mask].sum())
            st_total = int(mask.sum())
            station_dist.append({
                "station": st_id,
                "name": st_name,
                "anomalies": st_anom,
                "total": st_total,
                "rate_pct": round((st_anom / st_total) * 100, 2) if st_total > 0 else 0.0,
                "share_pct": round((st_anom / max(1, n_anomalies)) * 100, 2)
            })

        # Top 20 anomalous observations
        anom_indices = np.where(labels == 1)[0]
        sorted_anom_indices = anom_indices[np.argsort(scores[anom_indices])][:20]
        top_anomalies = []
        for rank, idx in enumerate(sorted_anom_indices, start=1):
            row = df.iloc[idx]
            st_code = str(row["STATION"])
            top_anomalies.append({
                "rank": rank,
                "station": st_code,
                "station_name": st_names.get(st_code, st_code),
                "date": str(row["DATE"]),
                "tmax": round(float(row["TMAX"]), 1),
                "tmin": round(float(row["TMIN"]), 1),
                "temp_range": round(float(row["TEMP_RANGE"]), 1),
                "prcp": round(float(row["PRCP"]), 1),
                "snow": round(float(row.get("SNOW", 0.0)), 1),
                "snwd": round(float(row.get("SNWD", 0.0)), 1),
                "score": round(float(scores[idx]), 4),
                "prediction": "Anomaly"
            })

        payload = {
            "algorithm": "Isolation Forest",
            "parameters": {
                "contamination": contamination,
                "n_estimators": n_estimators,
                "max_samples": max_samples,
                "random_state": random_state,
                "features_used": [f.replace("_SCALED", "") for f in selected_features]
            },
            "results": {
                "total_observations": len(labels),
                "anomalies_detected": n_anomalies,
                "normal_observations": n_normal,
                "anomaly_percentage": pct_anomalies,
                "score_min": round(float(scores.min()), 4),
                "score_median": round(float(np.median(scores)), 4),
                "score_max": round(float(scores.max()), 4),
                "score_threshold_zero": 0.0
            },
            "histogram": {
                "bin_centers": bin_centers,
                "frequencies": [int(x) for x in hist_counts],
                "normal_counts": [int(x) for x in hist_norm],
                "anomaly_counts": [int(x) for x in hist_anom]
            },
            "station_distribution": station_dist,
            "top_anomalies": top_anomalies,
            "execution": {
                "cached": was_cached,
                "duration_ms": duration_ms
            }
        }

        # Cache dynamic execution result
        self._if_dynamic_cache[cache_key] = payload
        return payload

    def run_local_outlier_factor(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.03,
        metric: str = "minkowski",
        features: Optional[list] = None
    ) -> Dict[str, Any]:
        """Execute Local Outlier Factor with input validation and runtime benchmarking.
        
        If n_neighbors is in precomputed set [10, 20, 30] at c=0.03 with default features,
        returns verified baseline metrics instantly. Otherwise, runs batch LOF.
        """
        if not (5 <= n_neighbors <= 100):
            raise ValueError(f"n_neighbors must be between 5 and 100; got {n_neighbors}")
        if not (0.005 <= contamination <= 0.20):
            raise ValueError(f"Contamination must be between 0.005 and 0.20; got {contamination}")
        if metric not in ("minkowski", "euclidean", "manhattan"):
            raise ValueError(f"Invalid metric '{metric}'. Supported metrics: minkowski, euclidean, manhattan")

        if features is None or len(features) == 0:
            selected_features = self._scaled_features
        else:
            selected_features = []
            for f in features:
                clean_f = f.replace("_SCALED", "")
                if clean_f not in NUMERICAL_FEATURE_COLS:
                    raise ValueError(f"Invalid feature '{f}'. Valid features: {NUMERICAL_FEATURE_COLS}")
                selected_features.append(f"{clean_f}_SCALED")

        cache_key = f"LOF_{n_neighbors}_{round(contamination, 3)}_{metric}_{','.join(sorted(selected_features))}"
        if cache_key in self._lof_dynamic_cache:
            cached_result = dict(self._lof_dynamic_cache[cache_key])
            cached_result["execution"] = dict(cached_result["execution"])
            cached_result["execution"]["cached"] = True
            return cached_result

        df = self.data_service.get_raw_consensus_dataframe()
        precomputed_cols = {
            10: ("LOF_ANOMALY_K10", "LOF_SCORE_K10", -1.3559),
            20: ("LOF_ANOMALY", "LOF_SCORE", -1.3632),
            30: ("LOF_ANOMALY_K30", "LOF_SCORE_K30", -1.3770)
        }

        t0 = time.perf_counter()
        is_baseline = (
            n_neighbors in precomputed_cols
            and round(contamination, 2) == 0.03
            and metric in ("minkowski", "euclidean")
            and selected_features == self._scaled_features
        )

        if is_baseline:
            anom_col, score_col, default_offset = precomputed_cols[n_neighbors]
            labels = df[anom_col].values
            scores = df[score_col].values
            was_cached = True
            offset_val = default_offset
        else:
            X_scaled = df[selected_features].values
            labels, scores, model = train_and_predict_lof(
                X_scaled,
                n_neighbors=n_neighbors,
                contamination=contamination
            )
            was_cached = False
            offset_val = float(model.offset_)

        duration_ms = round((time.perf_counter() - t0) * 1000, 2)

        n_anomalies = int(labels.sum())
        n_normal = len(labels) - n_anomalies
        pct_anomalies = round((n_anomalies / len(labels)) * 100, 2)

        # Histogram: 30 bins
        hist_counts, bin_edges = np.histogram(scores, bins=30)
        bin_centers = [round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 4) for i in range(len(hist_counts))]
        anom_mask = (labels == 1)
        norm_mask = (labels == 0)
        hist_anom, _ = np.histogram(scores[anom_mask], bins=bin_edges)
        hist_norm, _ = np.histogram(scores[norm_mask], bins=bin_edges)

        st_names = {
            "USW00094846": "Chicago O'Hare, IL",
            "USW00012839": "Miami Intl Airport, FL",
            "USW00094728": "NY Central Park, NY",
            "USW00024233": "Seattle Tacoma, WA"
        }
        station_dist = []
        for st_id, st_name in st_names.items():
            mask = (df["STATION"] == st_id)
            st_anom = int(labels[mask].sum())
            st_total = int(mask.sum())
            station_dist.append({
                "station": st_id,
                "name": st_name,
                "anomalies": st_anom,
                "total": st_total,
                "rate_pct": round((st_anom / st_total) * 100, 2) if st_total > 0 else 0.0,
                "share_pct": round((st_anom / max(1, n_anomalies)) * 100, 2)
            })

        # Top 20 anomalous observations
        anom_indices = np.where(labels == 1)[0]
        sorted_anom_indices = anom_indices[np.argsort(scores[anom_indices])][:20]
        top_anomalies = []
        for rank, idx in enumerate(sorted_anom_indices, start=1):
            row = df.iloc[idx]
            st_code = str(row["STATION"])
            top_anomalies.append({
                "rank": rank,
                "station": st_code,
                "station_name": st_names.get(st_code, st_code),
                "date": str(row["DATE"]),
                "tmax": round(float(row["TMAX"]), 1),
                "tmin": round(float(row["TMIN"]), 1),
                "temp_range": round(float(row["TEMP_RANGE"]), 1),
                "prcp": round(float(row["PRCP"]), 1),
                "snow": round(float(row.get("SNOW", 0.0)), 1),
                "snwd": round(float(row.get("SNWD", 0.0)), 1),
                "score": round(float(scores[idx]), 4),
                "prediction": "Anomaly"
            })

        payload = {
            "algorithm": "Local Outlier Factor",
            "parameters": {
                "n_neighbors": n_neighbors,
                "contamination": contamination,
                "novelty": False,
                "metric": metric,
                "features_used": [f.replace("_SCALED", "") for f in selected_features]
            },
            "results": {
                "total_observations": len(labels),
                "anomalies_detected": n_anomalies,
                "normal_observations": n_normal,
                "anomaly_percentage": pct_anomalies,
                "negative_outlier_factor_min": round(float(scores.min()), 4),
                "negative_outlier_factor_median": round(float(np.median(scores)), 4),
                "negative_outlier_factor_max": round(float(scores.max()), 4),
                "decision_offset_threshold": round(offset_val, 4)
            },
            "histogram": {
                "bin_centers": bin_centers,
                "frequencies": [int(x) for x in hist_counts],
                "normal_counts": [int(x) for x in hist_norm],
                "anomaly_counts": [int(x) for x in hist_anom]
            },
            "station_distribution": station_dist,
            "top_anomalies": top_anomalies,
            "execution": {
                "cached": was_cached,
                "duration_ms": duration_ms
            }
        }

        self._lof_dynamic_cache[cache_key] = payload
        return payload

