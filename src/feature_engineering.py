"""Feature Engineering Module for Weather-Station Observations.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Extracts physical weather differences, diurnal temperature range,
cyclical seasonal encodings, and temporal shifts.
"""

import numpy as np
import pandas as pd


def compute_physical_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive meteorologically meaningful physical features from core variables.
    
    Features created:
    - TEMP_RANGE: Diurnal Temperature Range (TMAX - TMIN). Key physical metric;
      negative values represent sensor swap errors, extreme values flag sensor freeze/runaway.
    - TEMP_MEAN: Daily midpoint temperature (TMAX + TMIN) / 2.
    - PRCP_SNOW_RATIO: Precipitation to Snowfall ratio (where snow > 0).
    - HAS_SNOW: Binary indicator if snow was present.
    - HAS_PRCP: Binary indicator if precipitation occurred.
    """
    data = df.copy()
    
    # Diurnal temperature range (°C)
    data["TEMP_RANGE"] = data["TMAX"] - data["TMIN"]
    
    # Midpoint mean temperature (°C)
    data["TEMP_MEAN"] = (data["TMAX"] + data["TMIN"]) / 2.0
    
    # Binary presence indicators
    data["HAS_PRCP"] = (data["PRCP"] > 0).astype(int)
    data["HAS_SNOW"] = (data["SNOW"] > 0).astype(int)
    
    return data


def compute_temporal_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract cyclical temporal representations for seasonal climate modeling.
    
    Weather is fundamentally seasonal. Representing day of year via sine and cosine
    preserves the continuous cyclical loop (December 31 is adjacent to January 1).
    """
    data = df.copy()
    
    if not np.issubdtype(data["DATE"].dtype, np.datetime64):
        data["DATE"] = pd.to_datetime(data["DATE"])
        
    data["YEAR"] = data["DATE"].dt.year
    data["MONTH"] = data["DATE"].dt.month
    data["DAY"] = data["DATE"].dt.day
    data["DAY_OF_YEAR"] = data["DATE"].dt.dayofyear
    
    # Meteorological season:
    # Winter: Dec (12), Jan (1), Feb (2) -> 1
    # Spring: Mar (3), Apr (4), May (5) -> 2
    # Summer: Jun (6), Jul (7), Aug (8) -> 3
    # Fall:   Sep (9), Oct (10), Nov (11) -> 4
    season_map = {
        12: 1, 1: 1, 2: 1,
        3: 2, 4: 2, 5: 2,
        6: 3, 7: 3, 8: 3,
        9: 4, 10: 4, 11: 4
    }
    data["SEASON"] = data["MONTH"].map(season_map)
    
    # Cyclical day-of-year transformations
    days_in_year = 365.25
    data["SIN_DOY"] = np.sin(2 * np.pi * data["DAY_OF_YEAR"] / days_in_year)
    data["COS_DOY"] = np.cos(2 * np.pi * data["DAY_OF_YEAR"] / days_in_year)
    
    return data


def compute_temporal_differencing(df: pd.DataFrame) -> pd.DataFrame:
    """Compute day-to-day rate-of-change features within each station.
    
    Sudden overnight spikes in temperature (e.g. +25°C in 24h) frequently highlight
    intermittent electrical glitches or equipment disconnects.
    """
    data = df.copy()
    data = data.sort_values(["STATION", "DATE"]).reset_index(drop=True)
    
    # 1-day differences grouped by station
    data["TMAX_DIFF_1D"] = data.groupby("STATION")["TMAX"].diff().fillna(0.0)
    data["TMIN_DIFF_1D"] = data.groupby("STATION")["TMIN"].diff().fillna(0.0)
    data["PRCP_DIFF_1D"] = data.groupby("STATION")["PRCP"].diff().fillna(0.0)
    
    return data


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Execute end-to-end feature engineering pipeline."""
    df = compute_physical_weather_features(df)
    df = compute_temporal_cyclical_features(df)
    df = compute_temporal_differencing(df)
    return df

