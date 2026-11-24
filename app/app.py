"""Interactive Streamlit Dashboard for Faulty Weather-Station Anomaly Detection.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Demonstrates end-to-end unsupervised detection pipeline:
Data Exploration, Isolation Forest, Local Outlier Factor, PCA 2D Visualization,
Algorithm Comparison, Historical Event Auditing, and Limitations.
"""

import os
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Streamlit Page Config
st.set_page_config(
    page_title="NOAA Weather Anomaly Detection (SIH26073 / INT396)",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths to data and results
ROOT_DIR = PROJECT_DIR
RESULTS_FILE = ROOT_DIR / "outputs" / "anomaly_results" / "consensus_anomaly_results.csv"
COMPARISON_FILE = ROOT_DIR / "outputs" / "comparison_tables" / "model_comparison_matrix.csv"
LOADINGS_FILE = ROOT_DIR / "outputs" / "comparison_tables" / "pca_component_loadings.csv"
EVENTS_FILE = ROOT_DIR / "outputs" / "evaluation_results" / "historical_extreme_events_audit.csv"
STATION_BIAS_FILE = ROOT_DIR / "outputs" / "evaluation_results" / "geographical_station_bias.csv"
SEASON_BIAS_FILE = ROOT_DIR / "outputs" / "evaluation_results" / "seasonal_bias.csv"
TOP_SUSPICIOUS_FILE = ROOT_DIR / "outputs" / "evaluation_results" / "top_suspicious_observations.csv"
PREPROC_SUMMARY_FILE = ROOT_DIR / "outputs" / "evaluation_results" / "preprocessing_summary.json"


@st.cache_data
def load_all_datasets():
    """Load cached experimental outputs."""
    df_results = pd.read_csv(RESULTS_FILE)
    df_results["DATE"] = pd.to_datetime(df_results["DATE"])
    
    df_comp = pd.read_csv(COMPARISON_FILE) if COMPARISON_FILE.exists() else pd.DataFrame()
    df_loadings = pd.read_csv(LOADINGS_FILE) if LOADINGS_FILE.exists() else pd.DataFrame()
    df_events = pd.read_csv(EVENTS_FILE) if EVENTS_FILE.exists() else pd.DataFrame()
    df_st_bias = pd.read_csv(STATION_BIAS_FILE) if STATION_BIAS_FILE.exists() else pd.DataFrame()
    df_se_bias = pd.read_csv(SEASON_BIAS_FILE) if SEASON_BIAS_FILE.exists() else pd.DataFrame()
    df_top = pd.read_csv(TOP_SUSPICIOUS_FILE) if TOP_SUSPICIOUS_FILE.exists() else pd.DataFrame()
    
    with open(PREPROC_SUMMARY_FILE, "r") as f:
        preproc_meta = json.load(f)
        
    return df_results, df_comp, df_loadings, df_events, df_st_bias, df_se_bias, df_top, preproc_meta


# Load datasets
try:
    df_main, df_comparison, df_pca_loadings, df_extreme_events, df_station_bias, df_season_bias, df_top_suspicious, preproc_info = load_all_datasets()
except Exception as e:
    st.error(f"Error loading experimental results: {e}. Please ensure Phases 2-8 have executed.")
    st.stop()


# ---------------------------------------------------------
# Sidebar Navigation & Station Filter
# ---------------------------------------------------------
st.sidebar.title("🌦️ Weather Anomaly Lab")
st.sidebar.caption("Course: **INT396 (Unsupervised ML)**")
st.sidebar.caption("Problem ID: **SIH26073**")

menu_selection = st.sidebar.radio(
    "Navigate Modules:",
    [
        "1. Project Overview",
        "2. Dataset Information",
        "3. Exploratory Data Analysis",
        "4. Preprocessing & Audit",
        "5. Isolation Forest Results",
        "6. Local Outlier Factor (LOF)",
        "7. Algorithm Comparison",
        "8. PCA 2D Visualization",
        "9. Anomaly Investigation & Events",
        "10. Scientific Limitations & Bias"
    ]
)

st.sidebar.markdown("---")
selected_station = st.sidebar.selectbox(
    "Filter by Station:",
    options=["All Stations"] + list(df_main["STATION"].unique()),
    index=0
)

# Apply station filter if selected
if selected_station != "All Stations":
    filtered_df = df_main[df_main["STATION"] == selected_station]
else:
    filtered_df = df_main


# ---------------------------------------------------------
# View 1: Project Overview
# ---------------------------------------------------------
if menu_selection == "1. Project Overview":
    st.title("Faulty Weather-Station Observation Detection")
    st.subheader("Unsupervised Anomaly Detection using Isolation Forest, LOF & PCA (SIH26073)")
    
    st.markdown("""
    ### Executive Summary
    Weather stations provide critical foundational observations for weather forecasting, climate modeling, and disaster preparedness.
    However, collected ground observations frequently contain anomalous readings due to **sensor degradation, electrical power fluctuations,
    calibration drifts, and human logging inconsistencies**—alongside **genuine extreme meteorological events** (such as blizzards, polar vortex freezes, and hurricane rainstorms).

    This university-level machine learning project demonstrates an unsupervised quality control system that identifies and explains statistically
    unusual observations across multi-year NOAA GHCN-Daily weather records using two complementary algorithms:
    - **Isolation Forest:** Tree-based ensemble isolating global multivariate outliers via recursive random partitioning.
    - **Local Outlier Factor (LOF):** Local density reachability estimating isolated points with substantially lower density than neighboring points.
    - **Principal Component Analysis (PCA):** 2D orthogonal projection visualizing consensus vs method-specific anomalies.
    """)
    
    # High-level Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observations", f"{len(df_main):,}")
    c2.metric("NOAA Stations", f"{df_main['STATION'].nunique()}")
    c3.metric("Baseline Contamination", "3.0%")
    c4.metric("Consensus Anomalies", f"{(df_main['CONSENSUS_LABEL'] == 'Consensus Anomaly').sum():,}")
    
    st.markdown("---")
    st.markdown("### Methodological Architecture")
    st.info("""
    **Pipeline Steps:**
    1. **NOAA Ingestion:** Decode tenths-of-unit scales (0.1°C, 0.1mm) and sentinel missing values (`-9999`).
    2. **Physical Rule Audit:** Validate physical constraints ($TMIN \le TMAX$, $PRCP \ge 0$).
    3. **Feature Engineering:** Compute diurnal range ($TMAX - TMIN$), cyclical seasonal encodings, and 1-day temporal differentials.
    4. **Unsupervised Modeling:** Run Isolation Forest ($c \in \{0.01, 0.03, 0.05\}$) and LOF ($k \in \{10, 20, 30\}$).
    5. **Consensus & PCA Projection:** Compute Jaccard similarity, agreement rates, and project onto 2D PCA plane.
    6. **Responsible Evaluation:** Differentiate verified historical climate extremes from potential sensor errors.
    """)


# ---------------------------------------------------------
# View 2: Dataset Information
# ---------------------------------------------------------
elif menu_selection == "2. Dataset Information":
    st.title("NOAA GHCN-Daily Dataset Specifications")
    st.markdown("""
    The primary data source is the official **NOAA Global Historical Climatology Network - Daily (GHCN-Daily)** archive maintained by NOAA NCEI.
    """)
    
    st.markdown("#### Station Metadata & Climate Zones")
    st_meta = df_main.groupby(["STATION", "NAME"]).agg(
        Latitude=("LATITUDE", "first"),
        Longitude=("LONGITUDE", "first"),
        Elevation_m=("ELEVATION", "first"),
        Start_Date=("DATE", "min"),
        End_Date=("DATE", "max"),
        Total_Records=("DATE", "count")
    ).reset_index()
    st.dataframe(st_meta, use_container_width=True)
    
    st.markdown("#### Core Meteorological Observations")
    st.markdown("""
    | Field | GHCN-Daily Raw Unit | Cleaned Metric Unit | Physical Role & Sensor Implications |
    | :--- | :--- | :--- | :--- |
    | `TMAX` | Tenths of °C | °C | Daily maximum temperature measured by automated electronic thermometer. |
    | `TMIN` | Tenths of °C | °C | Daily minimum temperature; must satisfy physical constraint $TMIN \le TMAX$. |
    | `TEMP_RANGE` | Calculated | °C | Diurnal temperature range ($TMAX - TMIN$). Inverted or extreme values flag sensor failures. |
    | `PRCP` | Tenths of mm | mm | Daily liquid precipitation measured by tipping-bucket or optical rain gauge. |
    | `SNOW` | Millimeters | mm | Daily snowfall accumulation. |
    | `SNWD` | Millimeters | mm | Ground snow depth measured by ultrasonic or acoustic snow depth sensors. |
    """)
    
    st.markdown("#### Raw Observation Sample")
    st.dataframe(df_main[["STATION", "DATE", "TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]].head(10), use_container_width=True)


# ---------------------------------------------------------
# View 3: Exploratory Data Analysis
# ---------------------------------------------------------
elif menu_selection == "3. Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    
    tab1, tab2, tab3 = st.tabs(["Distributions", "Diurnal Range vs Temperature", "Feature Correlations"])
    
    with tab1:
        st.subheader("Daily Maximum Temperature (°C) Distribution")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for station, grp in filtered_df.groupby("STATION"):
            sns.kdeplot(grp["TMAX"], ax=ax, label=station, fill=True, alpha=0.2)
        ax.set_title("KDE Density Distribution of Daily Maximum Temperature", fontweight="bold")
        ax.set_xlabel("TMAX (°C)")
        ax.legend(title="Station")
        st.pyplot(fig)
        plt.close(fig)
        
    with tab2:
        st.subheader("Diurnal Temperature Range vs. Maximum Temperature")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.scatterplot(
            data=filtered_df,
            x="TMAX",
            y="TEMP_RANGE",
            hue="STATION",
            alpha=0.6,
            s=25,
            ax=ax
        )
        ax.axhline(0, color="crimson", linestyle="--", label="Physical Bound (TMIN > TMAX)")
        ax.set_title("Diurnal Range (TMAX - TMIN) vs. Daily TMAX", fontweight="bold")
        ax.set_ylabel("Diurnal Range (°C)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        st.pyplot(fig)
        plt.close(fig)
        
    with tab3:
        st.subheader("Numerical Feature Correlation Matrix")
        feature_cols = preproc_info.get("numerical_features", ["TMAX", "TMIN", "TEMP_RANGE", "PRCP", "SNOW", "SNWD"])
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = filtered_df[feature_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------
# View 4: Preprocessing & Audit
# ---------------------------------------------------------
elif menu_selection == "4. Preprocessing & Audit":
    st.title("Data Preprocessing & Physical Quality Audit")
    
    st.markdown("""
    ### Preprocessing Protocol
    1. **NOAA Sentinel Handling:** All occurrences of `-9999` replaced with `np.nan`.
    2. **Metric Unit Normalization:** `TMAX`, `TMIN`, and `PRCP` scaled from tenths-of-units to standard SI units.
    3. **Missing Value Imputation:**
       - Tropical/arid snow depths filled with `0.0 mm`.
       - Temperatures imputed using temporal forward-fill preserving daily autocorrelation.
    4. **Feature Engineering:**
       - Diurnal range: $TEMP\_RANGE = TMAX - TMIN$.
       - Cyclical Day-of-Year: $\sin(2\pi \cdot DOY / 365.25)$ and $\cos(2\pi \cdot DOY / 365.25)$.
       - 1-Day Temporal Differences: $\Delta TMAX$, $\Delta TMIN$.
    5. **Feature Scaling:** `StandardScaler` applied without trimming outlier tails.
    """)
    
    st.markdown("#### Physical Quality Audit Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Inverted Temperatures (TMIN > TMAX)", "0")
    c2.metric("Negative Precipitation (PRCP < 0)", "0")
    c3.metric("Total Physical Constraint Violations", "0")
    
    st.markdown("#### Summary of Preprocessed Numerical Features")
    st.dataframe(filtered_df[preproc_info.get("numerical_features", [])].describe().round(2).T, use_container_width=True)


# ---------------------------------------------------------
# View 5: Isolation Forest Results
# ---------------------------------------------------------
elif menu_selection == "5. Isolation Forest Results":
    st.title("🌲 Isolation Forest Anomaly Detection")
    st.markdown("""
    **Isolation Forest** isolates observations by randomly selecting a feature and randomly splitting between the minimum and maximum values.
    Because anomalies possess unusual attribute values, they require substantially **fewer recursive partitions** (shorter average tree path length) to isolate.
    """)
    
    col_c = st.selectbox(
        "Select Contamination Parameter (α):",
        options=[0.01, 0.03, 0.05],
        index=1
    )
    c_suffix = f"C{int(col_c*100):02d}"
    
    anom_col = f"IF_ANOMALY_{c_suffix}"
    score_col = f"IF_SCORE_{c_suffix}"
    
    n_anom = int(filtered_df[anom_col].sum())
    n_norm = len(filtered_df) - n_anom
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Anomalies Flagged (c={col_c})", f"{n_anom:,}")
    c2.metric("Normal Observations", f"{n_norm:,}")
    c3.metric("Anomaly Proportion", f"{(n_anom / len(filtered_df))*100:.2f}%")
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    norm_pts = filtered_df[filtered_df[anom_col] == 0]
    anom_pts = filtered_df[filtered_df[anom_col] == 1]
    
    ax.scatter(norm_pts["TMIN"], norm_pts["TMAX"], c="lightgray", label="Normal", alpha=0.35, s=20)
    ax.scatter(anom_pts["TMIN"], anom_pts["TMAX"], c="crimson", label=f"IF Anomaly (c={col_c})", alpha=0.85, s=35, edgecolors="black", linewidths=0.4)
    ax.set_title(f"Isolation Forest Anomaly Detection (Contamination = {col_c})", fontweight="bold")
    ax.set_xlabel("Minimum Temperature (°C)")
    ax.set_ylabel("Maximum Temperature (°C)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("#### Sample Detected Isolation Forest Anomalies")
    st.dataframe(
        anom_pts[["STATION", "DATE", "TMAX", "TMIN", "TEMP_RANGE", "PRCP", "SNOW", score_col]].head(10),
        use_container_width=True
    )


# ---------------------------------------------------------
# View 6: Local Outlier Factor (LOF) Results
# ---------------------------------------------------------
elif menu_selection == "6. Local Outlier Factor (LOF)":
    st.title("📍 Local Outlier Factor (LOF) Detection")
    st.markdown("""
    **Local Outlier Factor (LOF)** measures the local density deviation of an observation with respect to its $k$-nearest neighbors.
    An observation with a substantially lower local reachability density than its neighbors is identified as an outlier.
    """)
    
    col_k = st.selectbox(
        "Select Neighborhood Parameter (k):",
        options=[10, 20, 30],
        index=1
    )
    
    anom_col = f"LOF_ANOMALY_K{col_k}"
    score_col = f"LOF_SCORE_K{col_k}"
    
    n_anom = int(filtered_df[anom_col].sum())
    n_norm = len(filtered_df) - n_anom
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"LOF Anomalies (k={col_k})", f"{n_anom:,}")
    c2.metric("Normal Observations", f"{n_norm:,}")
    c3.metric("Anomaly Proportion", f"{(n_anom / len(filtered_df))*100:.2f}%")
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    norm_pts = filtered_df[filtered_df[anom_col] == 0]
    anom_pts = filtered_df[filtered_df[anom_col] == 1]
    
    ax.scatter(norm_pts["TMIN"], norm_pts["TMAX"], c="lightgray", label="Normal Density", alpha=0.35, s=20)
    ax.scatter(anom_pts["TMIN"], anom_pts["TMAX"], c="royalblue", label=f"LOF Anomaly (k={col_k})", alpha=0.85, s=35, edgecolors="black", linewidths=0.4)
    ax.set_title(f"LOF Anomaly Detection on TMIN vs TMAX Plane (k = {col_k})", fontweight="bold")
    ax.set_xlabel("Minimum Temperature (°C)")
    ax.set_ylabel("Maximum Temperature (°C)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("#### Sample Detected LOF Anomalies")
    st.dataframe(
        anom_pts[["STATION", "DATE", "TMAX", "TMIN", "TEMP_RANGE", "PRCP", "SNOW", score_col]].head(10),
        use_container_width=True
    )


# ---------------------------------------------------------
# View 7: Algorithm Comparison
# ---------------------------------------------------------
elif menu_selection == "7. Algorithm Comparison":
    st.title("⚖️ Comparative Analysis: Isolation Forest vs. LOF")
    
    st.markdown("""
    ### Model Consensus Breakdown (Baseline: IF c=0.03, LOF k=20)
    """)
    
    c_counts = filtered_df["CONSENSUS_LABEL"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consensus Anomalies", f"{c_counts.get('Consensus Anomaly', 0):,}")
    c2.metric("Isolation Forest Only", f"{c_counts.get('IF Only', 0):,}")
    c3.metric("LOF Only", f"{c_counts.get('LOF Only', 0):,}")
    c4.metric("Agreement Rate", "95.33%")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=c_counts.index, y=c_counts.values, palette="Set2", ax=ax)
    for i, v in enumerate(c_counts.values):
        ax.text(i, v + 20, f"{v:,}", ha="center", fontweight="bold")
    ax.set_title("Observation Classification by Algorithm Consensus", fontweight="bold")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("#### Cross-Parameter Sensitivity Matrix (Contamination vs. Neighbors)")
    st.dataframe(df_comparison, use_container_width=True)


# ---------------------------------------------------------
# View 8: PCA 2D Visualization
# ---------------------------------------------------------
elif menu_selection == "8. PCA 2D Visualization":
    st.title("🌐 Principal Component Analysis (PCA) 2D Projection")
    st.markdown("""
    **Principal Component Analysis (PCA)** is an unsupervised linear dimensionality reduction technique.
    It rotates the 10-dimensional scaled meteorological feature space into orthogonal axes maximizing data variance.
    - **PC1 (27.96% Variance):** Captures seasonal thermodynamic state (`TMAX`, `TMIN`, `COS_DOY`).
    - **PC2 (16.08% Variance):** Captures temporal volatility & diurnal swing (`TMAX_DIFF_1D`, `TEMP_RANGE`).
    - **Cumulative 2D Explained Variance: 44.05%**
    """)
    
    pca_view = st.radio(
        "Select Coloring Perspective in PCA Space:",
        ["Consensus & Divergence", "Isolation Forest Only", "LOF Only", "Weather Stations"],
        horizontal=True
    )
    
    fig, ax = plt.subplots(figsize=(9, 6.5))
    if pca_view == "Consensus & Divergence":
        palette = {"Normal": "#bdc3c7", "Consensus Anomaly": "#e74c3c", "IF Only": "#e67e22", "LOF Only": "#2980b9"}
        sns.scatterplot(data=filtered_df, x="PC1", y="PC2", hue="CONSENSUS_LABEL", palette=palette, alpha=0.75, s=30, ax=ax)
    elif pca_view == "Isolation Forest Only":
        sns.scatterplot(data=filtered_df[filtered_df["IF_ANOMALY"] == 0], x="PC1", y="PC2", c="#bdc3c7", alpha=0.3, s=20, ax=ax, label="Normal")
        sns.scatterplot(data=filtered_df[filtered_df["IF_ANOMALY"] == 1], x="PC1", y="PC2", c="#e74c3c", alpha=0.85, s=35, ax=ax, label="IF Anomaly")
    elif pca_view == "LOF Only":
        sns.scatterplot(data=filtered_df[filtered_df["LOF_ANOMALY"] == 0], x="PC1", y="PC2", c="#bdc3c7", alpha=0.3, s=20, ax=ax, label="Normal Density")
        sns.scatterplot(data=filtered_df[filtered_df["LOF_ANOMALY"] == 1], x="PC1", y="PC2", c="#2980b9", alpha=0.85, s=35, ax=ax, label="LOF Anomaly")
    else:
        sns.scatterplot(data=filtered_df, x="PC1", y="PC2", hue="STATION", alpha=0.5, s=25, ax=ax)
        
    ax.set_title(f"2D PCA Feature Space: {pca_view}", fontweight="bold")
    ax.set_xlabel("Principal Component 1 (28.0% Variance Explained)")
    ax.set_ylabel("Principal Component 2 (16.1% Variance Explained)")
    st.pyplot(fig)
    plt.close(fig)
    
    st.markdown("#### Principal Component Feature Loadings (Eigenvectors)")
    st.dataframe(df_pca_loadings, use_container_width=True)


# ---------------------------------------------------------
# View 9: Anomaly Investigation & Events
# ---------------------------------------------------------
elif menu_selection == "9. Anomaly Investigation & Events":
    st.title("🔍 Anomaly Investigation & Real-World Historical Events")
    
    st.markdown("""
    ### Critical Scientific Distinction: Statistical Anomaly ≠ Sensor Fault
    An unsupervised model flags observations that are **statistically unusual**. However, genuine extreme meteorological phenomena
    (e.g., polar vortex freezes, hurricanes, record heat domes) reside naturally in the distribution tails.
    """)
    
    st.markdown("#### Audit of Verified Historical Extreme Weather Events")
    st.dataframe(df_extreme_events, use_container_width=True)
    
    st.markdown("#### Top Consensus Anomalies for Expert Manual Review")
    st.dataframe(df_top_suspicious, use_container_width=True)
    
    # Download button for consensus anomaly results
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Filtered Dataset with Anomaly Flags (CSV)",
        data=csv_bytes,
        file_name="weather_anomaly_results.csv",
        mime="text/csv"
    )


# ---------------------------------------------------------
# View 10: Scientific Limitations & Bias
# ---------------------------------------------------------
elif menu_selection == "10. Scientific Limitations & Bias":
    st.title("⚠️ Scientific Limitations, Bias & Responsible AI")
    
    tab1, tab2 = st.tabs(["Geographical & Seasonal Bias", "Responsible Interpretation Guide"])
    
    with tab1:
        st.subheader("1. Geographical Station Bias")
        st.dataframe(df_station_bias, use_container_width=True)
        st.caption("Notice: Stations in high-variance continental climates (Chicago) produce far higher anomaly rates than mild maritime climates (Seattle, Miami).")
        
        st.subheader("2. Seasonal Bias")
        st.dataframe(df_season_bias, use_container_width=True)
        st.caption("Notice: Winter conditions produce wider dispersion tails (freezing extremes, blizzards) yielding 9.66% IF anomaly flags compared to only 0.17% in Summer.")
        
    with tab2:
        st.markdown("""
        ### Responsible Machine Learning Principles in Climate Science
        1. **Automated Triage, Not Autonomous Erasure:**
           - Unsupervised anomaly detection must serve as a *prioritization funnel* for human meteorological experts.
           - Data points must never be silently dropped or modified without multi-station cross-validation or physical sensor inspection.
        2. **Multi-Station Clustered Baselines:**
           - Pooling disparate climate zones (e.g., Miami tropical savanna vs. Chicago continental) into a single global model introduces baseline bias.
           - Recommended best practice: Train station-specific or regional-cluster models with local seasonal baselines.
        3. **Instrumental Limitations:**
           - Tipping-bucket rain gauges can freeze during snowstorms, artificially underreporting precipitation until thawing occurs.
           - Acoustic snow sensors can register acoustic interference during severe wind gusts.
        """)

