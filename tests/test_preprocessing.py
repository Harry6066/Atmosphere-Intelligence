"""Unit tests for weather data preprocessing and feature engineering."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_preprocessing import decode_noaa_units, audit_physical_consistency, NUMERICAL_FEATURE_COLS
from src.feature_engineering import compute_physical_weather_features, compute_temporal_cyclical_features


def test_decode_noaa_units():
    """Verify conversion from NOAA tenths-of-units and -9999 sentinel handling."""
    raw_df = pd.DataFrame({
        "TMAX": [250, -17, -9999],
        "TMIN": [150, -72, -9999],
        "PRCP": [191, 0, -9999],
        "SNOW": [0, 50, -9999],
        "SNWD": [0, 100, -9999]
    })
    decoded = decode_noaa_units(raw_df)
    
    # 250 tenths -> 25.0 °C
    assert decoded.loc[0, "TMAX"] == pytest.approx(25.0)
    # -17 tenths -> -1.7 °C
    assert decoded.loc[1, "TMAX"] == pytest.approx(-1.7)
    # 191 tenths -> 19.1 mm
    assert decoded.loc[0, "PRCP"] == pytest.approx(19.1)
    # -9999 replaced with NaN
    assert np.isnan(decoded.loc[2, "TMAX"])
    assert np.isnan(decoded.loc[2, "PRCP"])


def test_physical_consistency_audit():
    """Verify detection of inverted temperatures (TMIN > TMAX)."""
    df = pd.DataFrame({
        "TMAX": [20.0, 10.0],
        "TMIN": [10.0, 15.0],   # Second row has physical inversion (15 > 10)
        "PRCP": [0.0, 5.0],
        "SNWD": [0.0, 0.0]
    })
    audited, summary = audit_physical_consistency(df)
    
    assert summary["inverted_temp_count"] == 1
    assert audited.loc[0, "FLAG_INVERTED_TEMP"] == 0
    assert audited.loc[1, "FLAG_INVERTED_TEMP"] == 1


def test_feature_engineering_diurnal_and_cyclical():
    """Verify calculation of diurnal range and cyclical temporal encodings."""
    df = pd.DataFrame({
        "DATE": pd.to_datetime(["2023-01-01", "2023-07-02"]),
        "TMAX": [5.0, 30.0],
        "TMIN": [-2.0, 18.0],
        "PRCP": [0.0, 10.0],
        "SNOW": [0.0, 0.0],
        "SNWD": [0.0, 0.0]
    })
    df_phys = compute_physical_weather_features(df)
    df_temp = compute_temporal_cyclical_features(df_phys)
    
    # Diurnal range: 5 - (-2) = 7
    assert df_temp.loc[0, "TEMP_RANGE"] == pytest.approx(7.0)
    # Midpoint mean: (30 + 18) / 2 = 24
    assert df_temp.loc[1, "TEMP_MEAN"] == pytest.approx(24.0)
    
    # Cyclical day of year bounds (-1 <= sin/cos <= 1)
    assert -1.0 <= df_temp.loc[0, "SIN_DOY"] <= 1.0
    assert -1.0 <= df_temp.loc[0, "COS_DOY"] <= 1.0


def test_processed_dataset_shape():
    """Verify that processed dataset exists and contains expected columns."""
    proc_path = PROJECT_DIR / "data" / "processed" / "ghcnd_processed.csv"
    assert proc_path.exists(), "ghcnd_processed.csv is missing"
    
    df = pd.read_csv(proc_path, nrows=50)
    for col in NUMERICAL_FEATURE_COLS:
        assert col in df.columns, f"Feature column {col} missing"
        assert f"{col}_SCALED" in df.columns, f"Scaled column {col}_SCALED missing"

