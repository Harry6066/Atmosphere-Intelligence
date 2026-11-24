"""Data Ingestion and Parsing Module for NOAA GHCN-Daily Records.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Fetches, validates, and parses official NOAA Global Historical Climatology
Network - Daily (GHCN-Daily) station observations.
"""

import os
import sys
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s"
)
logger = logging.getLogger("data_loader")

# Base URLs for official NOAA GHCN-Daily data
NOAA_DAILY_ACCESS_URL = (
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access"
)
NOAA_DOCS_URL = (
    "https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily"
)

# Reference stations across diverse, well-documented climate zones
DEFAULT_STATIONS: Dict[str, Dict[str, str]] = {
    "USW00094728": {
        "name": "NY CITY CENTRAL PARK, NY US",
        "climate_zone": "Humid Continental / Maritime",
        "state": "NY"
    },
    "USW00094846": {
        "name": "CHICAGO O'HARE INTERNATIONAL AIRPORT, IL US",
        "climate_zone": "Humid Continental (Severe Winters)",
        "state": "IL"
    },
    "USW00024233": {
        "name": "SEATTLE TACOMA INTERNATIONAL AIRPORT, WA US",
        "climate_zone": "Marine West Coast (High Precipitation)",
        "state": "WA"
    },
    "USW00012839": {
        "name": "MIAMI INTERNATIONAL AIRPORT, FL US",
        "climate_zone": "Tropical Monsoon / Savanna",
        "state": "FL"
    }
}

CORE_OBSERVATIONS = ["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]


def get_project_root() -> Path:
    """Return the base path of the Faulty_Weather_Station_Detection project."""
    # Handle being called from project root or inside src/
    current = Path(__file__).resolve().parent
    if current.name == "src":
        return current.parent
    return current


def download_station_data(
    station_id: str,
    output_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False
) -> Path:
    """Download raw daily CSV file for a given station ID from NOAA NCEI.
    
    Parameters
    ----------
    station_id : str
        The 11-character GHCN-Daily station identifier (e.g. 'USW00094728').
    output_dir : Path or str, optional
        Target directory to save raw data. Defaults to 'data/raw'.
    force_download : bool, default False
        Whether to overwrite existing local raw files.
        
    Returns
    -------
    Path
        Path to the saved raw CSV file.
    """
    if output_dir is None:
        output_dir = get_project_root() / "data" / "raw"
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    target_file = output_dir / f"{station_id}.csv"
    
    if target_file.exists() and not force_download:
        logger.info("Raw file already exists at: %s", target_file)
        return target_file
        
    url = f"{NOAA_DAILY_ACCESS_URL}/{station_id}.csv"
    logger.info("Downloading NOAA data for %s from %s ...", station_id, url)
    
    headers = {"User-Agent": "Mozilla/5.0 (Academic Research; INT396 University Project)"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(target_file, "wb") as f_out:
            f_out.write(response.read())
        logger.info("Successfully downloaded %s (%.2f KB)", station_id, target_file.stat().st_size / 1024)
    except Exception as exc:
        logger.error("Failed to download data for station %s: %s", station_id, exc)
        raise RuntimeError(
            f"Could not download NOAA station data from {url}. Verify internet connectivity."
        ) from exc
        
    return target_file


def parse_attribute_flags(attr_series: pd.Series) -> pd.DataFrame:
    """Parse NOAA GHCN-Daily comma-delimited observation attribute flags.
    
    Format: 'MeasurementFlag,QualityFlag,SourceFlag,TimeOfObservation'
    Quality flags are critical ground truth references:
        ' ' / null: Passed quality check
        'D': Failed duplicate check
        'G': Failed gap check
        'I': Failed internal consistency check
        'K': Failed streak/frequent-value check
        'L': Failed check on length of multiday period
        'M': Failed megaconsistency check
        'N': Failed naught check
        'O': Failed climatological outlier check
        'R': Failed lagged range check
        'S': Failed spatial consistency check
        'T': Failed temporal consistency check
        'W': Failed temperature too warm for snow
        'X': Failed bounds check
    """
    split_cols = attr_series.fillna(",,,").astype(str).str.split(",", expand=True)
    # Ensure 4 columns
    for col_idx in range(4):
        if col_idx not in split_cols.columns:
            split_cols[col_idx] = ""
            
    parsed = pd.DataFrame({
        "mflag": split_cols[0].str.strip(),
        "qflag": split_cols[1].str.strip(),
        "sflag": split_cols[2].str.strip(),
    }, index=attr_series.index)
    return parsed


def load_raw_station_data(
    file_path: Union[str, Path],
    start_year: Optional[int] = 2015,
    end_year: Optional[int] = 2023
) -> pd.DataFrame:
    """Load and parse raw NOAA station CSV file into structured DataFrame.
    
    Parameters
    ----------
    file_path : Path or str
        Path to local raw NOAA CSV file.
    start_year : int, optional
        Filter records starting from this year (inclusive).
    end_year : int, optional
        Filter records up to this year (inclusive).
        
    Returns
    -------
    pd.DataFrame
        Structured tabular observation dataset with extracted quality flags.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
        
    logger.info("Parsing NOAA station records from %s ...", path.name)
    df = pd.read_csv(path, low_memory=False)
    
    # Standardize column names
    df.columns = [c.strip().upper() for c in df.columns]
    
    # Required metadata columns
    meta_cols = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME"]
    available_meta = [c for c in meta_cols if c in df.columns]
    
    # Ensure DATE is datetime
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])
    
    # Filter by date range if provided
    if start_year is not None:
        df = df[df["DATE"].dt.year >= start_year]
    if end_year is not None:
        df = df[df["DATE"].dt.year <= end_year]
        
    df = df.sort_values("DATE").reset_index(drop=True)
    
    # Extract weather observation elements and their quality flags
    extracted = df[available_meta].copy()
    
    for element in CORE_OBSERVATIONS:
        if element in df.columns:
            extracted[element] = pd.to_numeric(df[element], errors="coerce")
            attr_col = f"{element}_ATTRIBUTES"
            if attr_col in df.columns:
                flags = parse_attribute_flags(df[attr_col])
                extracted[f"{element}_QFLAG"] = flags["qflag"]
                extracted[f"{element}_MFLAG"] = flags["mflag"]
            else:
                extracted[f"{element}_QFLAG"] = ""
                extracted[f"{element}_MFLAG"] = ""
        else:
            extracted[element] = np.nan
            extracted[f"{element}_QFLAG"] = ""
            extracted[f"{element}_MFLAG"] = ""
            
    logger.info(
        "Loaded %d daily records for station %s (%s to %s)",
        len(extracted),
        extracted["STATION"].iloc[0] if len(extracted) > 0 else "UNKNOWN",
        start_year,
        end_year
    )
    return extracted


def acquire_and_build_dataset(
    station_ids: Optional[List[str]] = None,
    start_year: int = 2016,
    end_year: int = 2023,
    save_combined_raw: bool = True
) -> pd.DataFrame:
    """Download and assemble the multi-station raw dataset for the project.
    
    Parameters
    ----------
    station_ids : list of str, optional
        List of NOAA station IDs to fetch. Defaults to DEFAULT_STATIONS keys.
    start_year : int, default 2016
    end_year : int, default 2023
    save_combined_raw : bool, default True
    
    Returns
    -------
    pd.DataFrame
        Combined multi-station raw observation DataFrame.
    """
    if station_ids is None:
        station_ids = list(DEFAULT_STATIONS.keys())
        
    raw_dir = get_project_root() / "data" / "raw"
    sample_dir = get_project_root() / "data" / "sample"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    dfs = []
    for sid in station_ids:
        raw_csv = download_station_data(sid, output_dir=raw_dir)
        station_df = load_raw_station_data(raw_csv, start_year=start_year, end_year=end_year)
        dfs.append(station_df)
        
    combined = pd.concat(dfs, ignore_index=True)
    
    if save_combined_raw:
        out_path = raw_dir / "ghcnd_raw_combined.csv"
        combined.to_csv(out_path, index=False)
        logger.info("Saved combined raw dataset to %s (%d rows)", out_path, len(combined))
        
        # Save sample subset for quick offline tests
        sample_path = sample_dir / "ghcnd_sample.csv"
        sample_df = combined.groupby("STATION").head(100).reset_index(drop=True)
        sample_df.to_csv(sample_path, index=False)
        logger.info("Saved sample fixture to %s (%d rows)", sample_path, len(sample_df))
        
    return combined


if __name__ == "__main__":
    print("=" * 70)
    print("NOAA GHCN-Daily Data Ingestion Pipeline (INT396 / SIH26073)")
    print("=" * 70)
    df_raw = acquire_and_build_dataset(start_year=2016, end_year=2023)
    print(f"\nIngestion Complete! Total observations: {len(df_raw):,}")
    print("\nStations Summary:")
    print(df_raw.groupby(["STATION", "NAME"]).agg(
        Records=("DATE", "count"),
        Start_Date=("DATE", "min"),
        End_Date=("DATE", "max")
    ))
    print("\nFirst 3 Records:")
    print(df_raw[["STATION", "DATE", "TMAX", "TMIN", "PRCP", "SNOW"]].head(3))

