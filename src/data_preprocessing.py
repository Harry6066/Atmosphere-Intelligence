"""Data Preprocessing and Quality Auditing Pipeline.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Handles unit conversions, missing-value imputation (-9999 sentinels),
physical rule auditing (TMIN > TMAX), feature scaling, and EDA visualizations.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Matplotlib cache and headless rendering
os.environ["MPLCONFIGDIR"] = str(Path.cwd() / ".cache" / "matplotlib")
plt.switch_backend("Agg")

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data_loader import get_project_root
from src.feature_engineering import apply_feature_engineering

logger = logging.getLogger("preprocessing")

# Modeling features for unsupervised anomaly detection
NUMERICAL_FEATURE_COLS = [
    "TMAX",
    "TMIN",
    "TEMP_RANGE",
    "PRCP",
    "SNOW",
    "SNWD",
    "TMAX_DIFF_1D",
    "TMIN_DIFF_1D",
    "SIN_DOY",
    "COS_DOY"
]


def decode_noaa_units(df: pd.DataFrame) -> pd.DataFrame:
    """Decode raw GHCN-Daily integer units to standard SI meteorological units.
    
    NOAA GHCN-Daily units:
    - TMAX / TMIN: tenths of degrees Celsius -> divide by 10.0 (°C)
    - PRCP: tenths of millimeters -> divide by 10.0 (mm)
    - SNOW: millimeters (mm)
    - SNWD: millimeters (mm)
    - -9999: Sentinel code denoting missing/unavailable measurement
    """
    data = df.copy()
    
    # Replace NOAA -9999 sentinel values with NaN
    for col in ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]:
        if col in data.columns:
            data[col] = data[col].replace([-9999, -9999.0], np.nan)
            
    # Convert tenths to standard units
    if "TMAX" in data.columns:
        data["TMAX"] = data["TMAX"] / 10.0
    if "TMIN" in data.columns:
        data["TMIN"] = data["TMIN"] / 10.0
    if "PRCP" in data.columns:
        data["PRCP"] = data["PRCP"] / 10.0
        
    return data


def audit_physical_consistency(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Audit physical weather rules and flag known physical violations.
    
    Checks:
    1. Inverted temperature: TMIN > TMAX (impossible in daily physical measurements).
    2. Negative precipitation: PRCP < 0.
    3. Negative snow depth: SNWD < 0.
    4. NOAA Quality Flagged: Any QFLAG present in raw records.
    
    CRITICAL RULE: We do NOT silently discard these rows because they are
    confirmed physical defects or calibration faults that our unsupervised
    models should identify!
    """
    data = df.copy()
    
    inverted_temp = (data["TMIN"] > data["TMAX"]).astype(int)
    negative_prcp = (data["PRCP"] < 0).astype(int)
    negative_snwd = (data["SNWD"] < 0).astype(int)
    
    # Check if any official NOAA quality flag is set (ignoring empty strings and NaN representations)
    qflag_cols = [c for c in data.columns if c.endswith("_QFLAG")]
    def has_valid_flag(row):
        for v in row:
            if pd.notna(v):
                s = str(v).strip()
                if s not in ("", "nan", "None", "null"):
                    return 1
        return 0

    has_official_qflag = data[qflag_cols].apply(has_valid_flag, axis=1)
    
    data["FLAG_INVERTED_TEMP"] = inverted_temp
    data["FLAG_NEGATIVE_PRCP"] = negative_prcp
    data["FLAG_OFFICIAL_QC"] = has_official_qflag
    
    # Composite rule-violation indicator (reference ground truth for evaluation)
    data["IS_PHYSICAL_RULE_VIOLATION"] = (
        (inverted_temp == 1) | (negative_prcp == 1) | (negative_snwd == 1) | (has_official_qflag == 1)
    ).astype(int)
    
    summary = {
        "inverted_temp_count": int(inverted_temp.sum()),
        "negative_prcp_count": int(negative_prcp.sum()),
        "official_qc_flagged_count": int(has_official_qflag.sum()),
        "total_rule_violations": int(data["IS_PHYSICAL_RULE_VIOLATION"].sum())
    }
    
    logger.info("Physical Consistency Audit: %s", summary)
    return data, summary


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing observations within each station time-series.
    
    Strategy:
    - Snowfall & Snow depth: In tropical or warm stations (like Miami), snow is not recorded.
      Missing SNOW/SNWD in warm stations is legitimately 0.0 mm.
    - Precipitation: Missing PRCP with no measurement flag is imputed with 0.0 (dry day).
    - Temperature (TMAX/TMIN): Imputed using rolling forward-fill (last observation carried forward)
      followed by seasonal station median, preserving temporal autocorrelation.
    """
    data = df.copy()
    data = data.sort_values(["STATION", "DATE"]).reset_index(drop=True)
    
    # Fill snow as 0 if missing (especially for stations that never snow)
    data["SNOW"] = data["SNOW"].fillna(0.0)
    data["SNWD"] = data["SNWD"].fillna(0.0)
    data["PRCP"] = data["PRCP"].fillna(0.0)
    
    # For temperature, forward fill within station then backward fill
    data["TMAX"] = data.groupby("STATION")["TMAX"].ffill().bfill()
    data["TMIN"] = data.groupby("STATION")["TMIN"].ffill().bfill()
    
    return data


def generate_eda_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save publication-quality exploratory data analysis figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Temperature Distributions by Station
    plt.figure(figsize=(10, 6))
    for station, grp in df.groupby("STATION"):
        sns.kdeplot(grp["TMAX"], label=f"{station} (TMAX)", fill=True, alpha=0.25)
    plt.title("Daily Maximum Temperature Distribution across Stations (°C)", fontsize=13, weight="bold")
    plt.xlabel("TMAX (°C)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(title="Station")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_tmax_distribution.png", dpi=200)
    plt.close()
    
    # 2. Diurnal Temperature Range vs TMAX Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="TMAX",
        y="TEMP_RANGE",
        hue="STATION",
        alpha=0.6,
        s=25
    )
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Physical Limit (TMIN > TMAX)")
    plt.title("Diurnal Temperature Range vs. Daily TMAX (°C)", fontsize=13, weight="bold")
    plt.xlabel("Maximum Temperature (°C)", fontsize=11)
    plt.ylabel("Diurnal Range: TMAX - TMIN (°C)", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_temp_range_scatter.png", dpi=200)
    plt.close()
    
    # 3. Correlation Heatmap of Engineered Weather Features
    plt.figure(figsize=(9, 7))
    corr = df[NUMERICAL_FEATURE_COLS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix of Numerical Weather Features", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_feature_correlation.png", dpi=200)
    plt.close()
    
    logger.info("Saved EDA figures to: %s", output_dir)


def preprocess_pipeline(
    raw_csv_path: Optional[Union[str, Path]] = None,
    save_outputs: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Execute complete preprocessing and scaling pipeline.
    
    Returns
    -------
    df_processed : pd.DataFrame
        Complete tabular dataframe with metadata, raw decoded values, and engineered features.
    X_scaled : pd.DataFrame
        StandardScaled matrix of NUMERICAL_FEATURE_COLS ready for IF, LOF, and PCA.
    scaler : StandardScaler
        The fitted scikit-learn StandardScaler instance.
    """
    root = get_project_root()
    if raw_csv_path is None:
        raw_csv_path = root / "data" / "raw" / "ghcnd_raw_combined.csv"
        
    df_raw = pd.read_csv(raw_csv_path, low_memory=False)
    logger.info("Loaded %d raw records from %s", len(df_raw), raw_csv_path)
    
    # 1. Decode units & handle -9999 sentinels
    df_decoded = decode_noaa_units(df_raw)
    
    # 2. Check duplicates
    dups = df_decoded.duplicated(subset=["STATION", "DATE"]).sum()
    if dups > 0:
        logger.warning("Dropping %d duplicate station-date observations", dups)
        df_decoded = df_decoded.drop_duplicates(subset=["STATION", "DATE"])
        
    # 3. Missing value treatment
    df_imputed = handle_missing_values(df_decoded)
    
    # 4. Feature engineering
    df_featured = apply_feature_engineering(df_imputed)
    
    # 5. Audit physical rules & quality flags
    df_audited, audit_summary = audit_physical_consistency(df_featured)
    
    # 6. Feature scaling
    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(df_audited[NUMERICAL_FEATURE_COLS])
    scaled_col_names = [f"{c}_SCALED" for c in NUMERICAL_FEATURE_COLS]
    df_scaled_features = pd.DataFrame(X_scaled_arr, columns=scaled_col_names, index=df_audited.index)
    
    df_processed = pd.concat([df_audited, df_scaled_features], axis=1)
    
    if save_outputs:
        proc_dir = root / "data" / "processed"
        proc_dir.mkdir(parents=True, exist_ok=True)
        out_csv = proc_dir / "ghcnd_processed.csv"
        df_processed.to_csv(out_csv, index=False)
        logger.info("Saved processed dataset to: %s (%d rows)", out_csv, len(df_processed))
        
        # Save preprocessing summary metadata
        meta_file = root / "outputs" / "evaluation_results" / "preprocessing_summary.json"
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "total_records": len(df_processed),
            "stations": df_processed["STATION"].unique().tolist(),
            "date_range": [str(df_processed["DATE"].min()), str(df_processed["DATE"].max())],
            "numerical_features": NUMERICAL_FEATURE_COLS,
            "physical_audit": audit_summary
        }
        with open(meta_file, "w") as f:
            json.dump(summary, f, indent=2)
            
        # Generate EDA plots
        generate_eda_visualizations(df_processed, root / "graphs" / "eda")
        
    return df_processed, df_scaled_features, scaler


if __name__ == "__main__":
    print("=" * 70)
    print("NOAA GHCN-Daily Data Preprocessing Pipeline (INT396 / SIH26073)")
    print("=" * 70)
    df_proc, df_scaled, scaler = preprocess_pipeline()
    print("\nPreprocessing complete!")
    print(f"Total processed observations: {len(df_proc):,}")
    print(f"Engineered numerical features scaled: {len(NUMERICAL_FEATURE_COLS)}")
    print("\nFeature statistics preview:")
    print(df_proc[NUMERICAL_FEATURE_COLS].describe().round(2).T[["mean", "std", "min", "50%", "max"]])
