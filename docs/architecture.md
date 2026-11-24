# System Architecture: Faulty Weather-Station Anomaly Detection

**Course:** INT396 – Unsupervised Machine Learning  
**Problem Statement ID:** SIH26073  
**Project Title:** Faulty Weather-Station Observation Detection Using Anomaly Detection Methods  

---

## 1. High-Level Architecture Diagram

```
+-----------------------------------------------------------------------------+
|                                DATA SOURCE                                  |
|         NOAA Global Historical Climatology Network - Daily (GHCN-Daily)     |
|         Official Access Archive (NCEI NOAA): Station Daily Records          |
+-----------------------------------------------------------------------------+
                                      │
                                      ▼
+-----------------------------------------------------------------------------+
|                         DATA INGESTION & AUDIT                              |
|                         (src/data_loader.py)                                |
|  - Parse fixed-width / CSV observation schemas                              |
|  - Decode tenths-of-unit scales (TMAX/TMIN 0.1°C, PRCP 0.1mm)              |
|  - Replace sentinel -9999 codes with NaN                                    |
|  - Extract Measurement, Quality (QFLAG), and Source flags                   |
+-----------------------------------------------------------------------------+
                                      │
                                      ▼
+-----------------------------------------------------------------------------+
|                    PREPROCESSING & FEATURE ENGINEERING                     |
|                 (src/data_preprocessing.py, feature_engineering.py)         |
|  - Autoregressive forward-fill & seasonal median imputation                 |
|  - Physical rule audit (TMIN > TMAX inverted temp checks)                   |
|  - Diurnal Temperature Range: TEMP_RANGE = TMAX - TMIN                      |
|  - Cyclical temporal transforms: sin(2π·DOY/365.25), cos(2π·DOY/365.25)    |
|  - 1-Day temporal difference: ΔTMAX, ΔTMIN                                  |
|  - Non-destructive feature scaling via StandardScaler                       |
+-----------------------------------------------------------------------------+
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
+------------------------------------+ +------------------------------------+
|         ISOLATION FOREST           | |       LOCAL OUTLIER FACTOR         |
|      (src/isolation_forest.py)     | |       (src/lof_detection.py)       |
|  - Global tree-based partitioning  | |  - Local density reachability      |
|  - Contaminations: 0.01, 0.03, 0.05| |  - Neighborhood sizes: 10, 20, 30  |
|  - Average path length score       | |  - Negative outlier factor scores  |
|  - Binary labels + decision scores | |  - Binary labels + local densities |
+------------------------------------+ +------------------------------------+
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
+-----------------------------------------------------------------------------+
|                         COMPARATIVE ANALYSIS ENGINE                         |
|                            (src/comparison.py)                              |
|  - Inter-model consensus anomalies (IF ∩ LOF)                               |
|  - Method-specific divergences (IF \ LOF, LOF \ IF)                         |
|  - Jaccard similarity index across parameter combinations                   |
|  - Agreement percentage and cross-parameter sensitivity matrix              |
+-----------------------------------------------------------------------------+
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
+------------------------------------+ +------------------------------------+
|          PCA VISUALIZATION         | |     EVALUATION & BIAS AUDIT        |
|        (src/pca_analysis.py)       | |        (src/evaluation.py)         |
|  - 2D orthogonal projection        | |  - Historical extreme event audit  |
|  - Variance explained: 44.05%      | |  - Statistical outlier vs defect   |
|  - Feature eigenvector loadings    | |  - Geographical & seasonal bias    |
+------------------------------------+ +------------------------------------+
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
+------------------------------------+ +------------------------------------+
|    INTERACTIVE STREAMLIT APP       | |   ACADEMIC REPORTS & ARTIFACTS     |
|           (app/app.py)             | |       (PDF, PPTX, Notebooks)       |
|  - Multi-tab visual dashboard      | |  - Formal university submission    |
|  - Dynamic filters & CSV downloads | |  - Verified experimental results   |
+------------------------------------+ +------------------------------------+
```

---

## 2. Component Descriptions

1. **`src/data_loader.py`**: Interacts with the NOAA NCEI archive to retrieve daily weather files, validate file structures, parse raw attribute flags, and save immutable raw datasets into `data/raw/`.
2. **`src/feature_engineering.py`**: Computes physical weather indicators including Diurnal Temperature Range ($TMAX - TMIN$), midpoint temperatures, cyclical trigonometric day-of-year features, and 1-day temporal rates of change.
3. **`src/data_preprocessing.py`**: Converts raw tenths-of-units, converts missing sentinels (`-9999`), carries out physical consistency audits ($TMIN \le TMAX$), and applies `StandardScaler` to prepare feature matrices.
4. **`src/isolation_forest.py`**: Trains Scikit-learn's `IsolationForest` across contamination rates $\alpha \in \{0.01, 0.03, 0.05\}$, logging continuous decision scores and binary anomaly flags into `outputs/anomaly_results/`.
5. **`src/lof_detection.py`**: Executes Scikit-learn's `LocalOutlierFactor` across neighborhood parameters $k \in \{10, 20, 30\}$, saving negative outlier factors and binary density anomaly labels.
6. **`src/comparison.py`**: Cross-references predictions between Isolation Forest and LOF, computing exact Jaccard similarity coefficients, overall agreement percentages, and consensus category assignments.
7. **`src/pca_analysis.py`**: Computes 2-component Principal Component Analysis (PCA) capturing 44.05% of dataset variance, generates component loading matrices, and plots 2D projections.
8. **`src/evaluation.py`**: Cross-validates flagged observations against known historical extreme events (2019 Polar Vortex, 2021 Pacific Northwest Heat Dome, Hurricane Ida), quantifies geographical station bias, and documents limitations.
9. **`app/app.py`**: Comprehensive interactive Streamlit web dashboard providing dynamic parameter tuning, data exploration, PCA projections, and downloadable anomaly audit reports.

