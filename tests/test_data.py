"""Unit tests for NOAA GHCN-Daily data loading and validation."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_loader import parse_attribute_flags, DEFAULT_STATIONS


def test_default_stations_definition():
    """Ensure standard climate stations are configured."""
    assert len(DEFAULT_STATIONS) >= 4
    assert "USW00094728" in DEFAULT_STATIONS
    assert "USW00094846" in DEFAULT_STATIONS


def test_attribute_flag_parsing():
    """Test decoding of NOAA attribute flags (Measurement, Quality, Source)."""
    raw_flags = pd.Series([
        ",,Z,",      # Blank measurement, blank quality, Z source
        "T,K,0,",    # Trace measurement, K quality failure, 0 source
        np.nan       # Missing attribute
    ])
    parsed = parse_attribute_flags(raw_flags)
    
    assert list(parsed.columns) == ["mflag", "qflag", "sflag"]
    assert parsed.iloc[0]["qflag"] == ""
    assert parsed.iloc[1]["qflag"] == "K"
    assert parsed.iloc[1]["mflag"] == "T"
    assert parsed.iloc[2]["qflag"] == ""


def test_raw_dataset_existence():
    """Verify that the raw combined dataset exists and has expected columns."""
    raw_path = PROJECT_DIR / "data" / "raw" / "ghcnd_raw_combined.csv"
    assert raw_path.exists(), "Raw dataset ghcnd_raw_combined.csv is missing"
    
    df = pd.read_csv(raw_path, nrows=50)
    expected_cols = ["STATION", "DATE", "TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]
    for col in expected_cols:
        assert col in df.columns, f"Expected column {col} missing in raw data"
