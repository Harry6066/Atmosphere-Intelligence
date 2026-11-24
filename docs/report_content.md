# University Project Report: Faulty Weather-Station Observation Detection Using Anomaly Detection Methods

**Course Code & Title:** INT396 – Unsupervised Machine Learning  
**Problem Statement ID:** SIH26073  
**Academic Year:** 2026  
**Curriculum Topic:** Unit IV & V – Anomaly Detection, Isolation Forest, Local Outlier Factor, and PCA Visualization  

---

## 1. Title Page

**Project Title:** Faulty Weather-Station Observation Detection Using Anomaly Detection Methods  
**Course Code:** INT396  
**Course Name:** Unsupervised Machine Learning  
**Problem Statement ID:** SIH26073  

**Group Members & Registration Details:**
- **Student 1:** [PENDING: Full Name] | Registration No: [PENDING: Registration Number]
- **Student 2:** [PENDING: Full Name] | Registration No: [PENDING: Registration Number]
- **Student 3:** [PENDING: Full Name] | Registration No: [PENDING: Registration Number]

**Faculty Instructor / Academic Mentor:**
- [PENDING: Instructor / Supervisor Name], Department of Computer Science & Engineering / Machine Learning

---

## 2. Introduction and Problem Statement

### 2.1 Introduction
Automated weather monitoring stations across the globe form the scientific backbone of numerical weather prediction, flood warnings, aviation routing, agricultural scheduling, and climate change research. Ground stations continuously log environmental variables such as maximum daily temperature ($TMAX$), minimum daily temperature ($TMIN$), liquid precipitation ($PRCP$), snowfall ($SNOW$), and snow depth ($SNWD$). 

However, surface weather sensors are exposed to extreme environmental wear, freezing conditions, electrical surges, lightning strikes, calibration drift, communication transmission packet loss, and transcription errors. Identifying bad data in real time is imperative to maintain the integrity of climate repositories.

### 2.2 Problem Statement (SIH26073)
Manual inspection of hundreds of thousands of continuous daily weather records across disparate geography is computationally and humanly infeasible. Furthermore, traditional static thresholding (such as rejecting temperatures $> 45^\circ\text{C}$) fails because acceptable ranges vary drastically between maritime, continental, and tropical climates.

Critically, a major scientific dilemma in meteorological quality control is **distinguishing between sensor malfunctions and genuine extreme weather events**: a historic polar vortex or hurricane produces extreme statistical outliers that represent real, valid meteorological phenomena rather than broken hardware.

This project implements an unsupervised machine learning architecture utilizing **Isolation Forest** and **Local Outlier Factor (LOF)**, combined with **Principal Component Analysis (PCA)** 2D projection, to detect, contrast, and explain anomalies without relying on artificial ground-truth labels.

---

## 3. Project Objectives and Need

### 3.1 Primary Objectives
1. **Acquire & Ingest NOAA Records:** Parse official NOAA Global Historical Climatology Network - Daily (GHCN-Daily) records with exact attribute flags.
2. **Handle Physical Rules & Missing Data:** Address sentinel missing value codes (`-9999`) and audit physical constraints ($TMIN \le TMAX$, $PRCP \ge 0$).
3. **Feature Engineering:** Derive diurnal temperature range ($TEMP\_RANGE = TMAX - TMIN$), cyclical seasonal encodings, and day-to-day rate of change.
4. **Implement Isolation Forest:** Partition feature space across contamination levels $\alpha \in \{0.01, 0.03, 0.05\}$.
5. **Implement Local Outlier Factor:** Compute local density reachabilities across neighborhood sizes $k \in \{10, 20, 30\}$.
6. **Comparative Consistency Analysis:** Compute consensus intersections ($IF \cap LOF$), Jaccard similarity matrices, and agreement percentages.
7. **2D PCA Projection:** Reduce 10-dimensional scaled space to 2 orthogonal principal components with explained variance and eigenvector loading analysis.
8. **Responsible Evaluation & Event Validation:** Cross-reference detections against documented historical climate extremes (2019 Polar Vortex, 2021 Heat Dome, Hurricane Ida) to separate statistical anomalies from sensor faults.
9. **Interactive Demonstration Dashboard:** Deliver an interactive multi-tab Streamlit web application.

---

## 4. Tools, Technologies, and Frameworks

- **Programming Language:** Python 3.11.15
- **Data Manipulation:** Pandas 2.2.3, NumPy 2.0.2
- **Scientific Computing:** SciPy 1.14.1
- **Machine Learning & PCA:** Scikit-learn 1.5.2 (`IsolationForest`, `LocalOutlierFactor`, `PCA`, `StandardScaler`)
- **Data Visualization:** Matplotlib 3.9.4, Seaborn 0.13.2
- **Web Application:** Streamlit 1.39.1
- **Automated Testing:** Pytest 8.2.2 (13 automated unit tests)
- **Document Generation:** FPDF2 2.9.0, Python-PPTX 1.0.2

---

## 5. Methodology and Implementation

### 5.1 Dataset Details
- **Source:** NOAA NCEI GHCN-Daily Archive
- **Stations Selected:**
  - `USW00094728`: New York City Central Park, NY (Humid Continental / Maritime)
  - `USW00094846`: Chicago O'Hare International Airport, IL (Continental / Severe Winters)
  - `USW00024233`: Seattle Tacoma International Airport, WA (Marine West Coast)
  - `USW00012839`: Miami International Airport, FL (Tropical Savanna)
- **Time Range:** January 1, 2016 – December 31, 2023 (8 full years = 2,922 days per station)
- **Total Ingested Observations:** 11,688 daily records

### 5.2 Preprocessing & Feature Engineering
- **Unit Conversion:** Converted tenths of degrees Celsius to $^\circ\text{C}$ and tenths of mm to $\text{mm}$.
- **Imputation:** Applied forward-fill followed by seasonal median within each station time-series.
- **Engineered Features:**
  - $TEMP\_RANGE = TMAX - TMIN$ (Physical stability check)
  - $\sin\left(\frac{2\pi \cdot DOY}{365.25}\right), \cos\left(\frac{2\pi \cdot DOY}{365.25}\right)$ (Cyclical seasonal envelope)
  - $\Delta TMAX_{1D}, \Delta TMIN_{1D}$ (Day-to-day rate of change)
- **Scaling:** Standardized using `StandardScaler` to zero mean and unit variance.

### 5.3 Modeling Implementations
- **Isolation Forest:** Tree-based ensemble with $n\_estimators=100$, random state 42. Evaluated at contamination rates 0.01, 0.03, and 0.05.
- **Local Outlier Factor (LOF):** Density-based reachability using Euclidean distance. Evaluated at neighborhood configurations $k=10, 20, 30$ with baseline contamination 0.03.

---

## 6. Experimental Results and Analysis

### 6.1 Anomaly Detection Summary (Total Observations: 11,688)

| Model Configuration | Anomalies Flagged | Percentage | Score Threshold / Offset |
| :--- | :--- | :--- | :--- |
| **Isolation Forest ($c=0.01$)** | 117 | 1.00% | 0.000 |
| **Isolation Forest ($c=0.03$)** | 351 | 3.00% | 0.000 |
| **Isolation Forest ($c=0.05$)** | 585 | 5.01% | 0.000 |
| **LOF ($k=10, c=0.03$)** | 351 | 3.00% | -1.356 |
| **LOF ($k=20, c=0.03$)** | 351 | 3.00% | -1.363 |
| **LOF ($k=30, c=0.03$)** | 351 | 3.00% | -1.377 |

### 6.2 Baseline Comparative Consensus ($IF_{c=0.03}$ vs $LOF_{k=20}$)
- **Consensus Anomalies ($IF \cap LOF$):** 78 observations
- **Isolation Forest Only ($IF \setminus LOF$):** 273 observations
- **Local Outlier Factor Only ($LOF \setminus IF$):** 273 observations
- **Total Union of Anomalies:** 624 observations
- **Agreement Percentage:** **95.33%**
- **Jaccard Similarity Index:** **0.1250**

### 6.3 Cross-Parameter Sensitivity Matrix

| IF Contamination ($c$) | LOF Neighbors ($k$) | Consensus Anomalies | Agreement % | Jaccard Similarity |
| :---: | :---: | :---: | :---: | :---: |
| 0.01 | 10 | 23 | 96.39% | 0.0517 |
| 0.01 | 20 | 30 | 96.51% | 0.0685 |
| 0.01 | 30 | 28 | 96.48% | 0.0636 |
| 0.03 | 10 | 54 | 94.92% | 0.0833 |
| **0.03** | **20** | **78** | **95.33%** | **0.1250** |
| 0.03 | 30 | 83 | 95.41% | 0.1341 |
| 0.05 | 10 | 87 | 93.48% | 0.1025 |
| 0.05 | 20 | 117 | 93.99% | 0.1429 |
| 0.05 | 30 | 132 | 94.25% | 0.1642 |

### 6.4 Principal Component Analysis (PCA) Results
- **PC1 Explained Variance:** **27.96%** (Thermal baseline: dominated by $TMAX$ loading +0.572, $TMIN$ +0.533, and winter cycle $\cos(DOY)$ -0.439)
- **PC2 Explained Variance:** **16.08%** (Dynamic instability: dominated by 1-day change $\Delta TMAX$ +0.662, $\Delta TMIN$ +0.444, and diurnal range $TEMP\_RANGE$ +0.432)
- **Cumulative 2D Explained Variance:** **44.05%**

---

## 7. Responsible Evaluation: Statistical Outlier vs. Sensor Fault

### 7.1 Historical Climate Events Case Studies
Our evaluation cross-referenced flagged anomalies against officially documented extreme meteorological events:
1. **Chicago Polar Vortex (Jan 29–31, 2019):** Chicago O'Hare recorded historic subzero lows reaching $-30.5^\circ\text{C}$ ($-23^\circ\text{F}$). Both Isolation Forest and LOF flagged **3 out of 3 days** as anomalies.
2. **Hurricane Ida Inundation (Sep 1–2, 2021):** Central Park recorded torrential rainfall $> 180\text{ mm}$ in 24 hours. Flagged by both models.
3. **Chicago Blizzard & Snowpack (Feb 15–16, 2021):** Snow depth reached $560\text{ mm}$. Flagged 2 out of 2 days by both models.
4. **Pacific Northwest Heat Dome (Jun 26–28, 2021):** Seattle recorded an all-time record high of $42.2^\circ\text{C}$ ($108^\circ\text{F}$). Flagged by LOF as a local density anomaly in coastal Washington.

**Scientific Finding:** These observations are **confirmed valid physical weather extremes**. They represent statistical outliers in the historical distribution, **NOT sensor malfunctions**. Unsupervised machine learning alone cannot determine causality; it provides automated candidate triage for meteorological experts.

### 7.2 Geographical and Seasonal Bias
- **Geographical Disparity:** Chicago O'Hare (continental climate) registered a 7.77% IF anomaly rate, whereas Miami (subtropical climate) registered only 0.24%. Pooling disparate climate regimes into a single global model causes milder regions to be under-flagged and variable regions to be over-flagged.
- **Seasonal Disparity:** Winter observations produced a 9.66% IF anomaly rate compared to only 0.17% in Summer, reflecting the higher variance of sub-zero freezes and snow accumulation.

---

## 8. Individual Contributions of Group Members

- **Member 1 ([PENDING: Student 1 Name]):**
  - Dataset acquisition pipeline from NOAA NCEI (`src/data_loader.py`).
  - Physical rule audit, unit decoding, and missing value treatment (`src/data_preprocessing.py`).
  - Unit tests for data loading and preprocessing (`tests/test_data.py`, `tests/test_preprocessing.py`).

- **Member 2 ([PENDING: Student 2 Name]):**
  - Feature engineering of diurnal range, cyclical calendar features, and 1-day temporal shifts (`src/feature_engineering.py`).
  - Isolation Forest model implementation and multi-parameter contamination experiments (`src/isolation_forest.py`).
  - Principal Component Analysis (PCA) projection, explained variance calculation, and loading analysis (`src/pca_analysis.py`).

- **Member 3 ([PENDING: Student 3 Name]):**
  - Local Outlier Factor (LOF) density-based reachability implementation (`src/lof_detection.py`).
  - Comparative analysis engine, Jaccard similarity, and parameter sensitivity matrices (`src/comparison.py`).
  - Interactive Streamlit dashboard development (`app/app.py`) and historical event validation (`src/evaluation.py`).

---

## 9. Limitations and Future Scope

### Limitations:
1. **Unsupervised Nature:** Unsupervised anomaly scores denote statistical rarity, not verified mechanical or electrical defects.
2. **Global Pooled Bias:** High-variance stations dominate outlier detection unless local regional grouping is applied.
3. **Instrumental Artifacts:** Rain gauge freeze-ups during blizzards can produce false zero-precipitation readings that pass basic consistency checks.

### Future Scope:
1. **Spatial Neighbor Verification:** Compare candidate anomalies against readings from adjacent stations within a 50 km radius (spatial cross-validation).
2. **Semi-Supervised Active Learning:** Allow meteorologists to confirm or dismiss flagged points via the Streamlit interface, iteratively refining isolation thresholds.
3. **Autoencoder & Graph Neural Networks:** Explore temporal graph neural networks modeling the physical topology of weather station sensor networks.

---

## 10. References and Resources

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. Eighth IEEE International Conference on Data Mining (ICDM), 413-422.
2. Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). *LOF: identifying density-based local outliers*. ACM SIGMOD Record, 29(2), 93-104.
3. Menne, M. J., Durre, I., Vose, R. S., Gleason, B. E., & Houston, T. G. (2012). *An overview of the Global Historical Climatology Network-Daily database*. Journal of Atmospheric and Oceanic Technology, 29(7), 897-910.
4. National Centers for Environmental Information (NCEI). NOAA GHCN-Daily Documentation. https://www.ncei.noaa.gov/
5. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.

