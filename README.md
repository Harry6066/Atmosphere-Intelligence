# ATMOSPHERE INTELLIGENCE: Faulty Weather-Station Observation Detection Using Anomaly Detection Methods

[![Course: INT396](https://img.shields.io/badge/Course-INT396%20Unsupervised%20ML-blue.svg)](file:///c:/Users/kalis/.gemini/antigravity-ide/scratch/New%20Project/Faulty_Weather_Station_Detection)
[![Problem ID: SIH26073](https://img.shields.io/badge/Problem%20ID-SIH26073-green.svg)](file:///c:/Users/kalis/.gemini/antigravity-ide/scratch/New%20Project/Faulty_Weather_Station_Detection)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11.15-brightgreen.svg)](https://www.python.org/)
[![Framework: Flask](https://img.shields.io/badge/Framework-Flask%203.1-black.svg)](https://flask.palletsprojects.com/)
[![Tests: 89 Passed](https://img.shields.io/badge/Tests-89%20Passed%20(100%25)-success.svg)](file:///c:/Users/kalis/.gemini/antigravity-ide/scratch/New%20Project/Faulty_Weather_Station_Detection/tests)

---

## 1. Project Title
**Faulty Weather-Station Observation Detection Using Anomaly Detection Methods**  
*Platform Operational Designation:* **ATMOSPHERE INTELLIGENCE**

---

## 2. Problem Statement ID
**SIH26073**  
*Course:* INT396 — Unsupervised Machine Learning  
*Domain:* Environmental Data Science & Climatological Quality Control

---

## 3. Project Overview
In environmental monitoring and operational meteorology, surface weather stations continuously measure and transmit primary climatic variables: Maximum Temperature (`TMAX`), Minimum Temperature (`TMIN`), Precipitation (`PRCP`), Snowfall (`SNOW`), and Snow Depth (`SNWD`). However, weather sensor networks routinely suffer from observation anomalies triggered by physical hardware malfunctions, sensor drift, electrical line degradation, icing, telemetry corruption, and transcription discrepancies—alongside authentic extreme meteorological events (such as cold waves, heat domes, and flash floods).

**ATMOSPHERE INTELLIGENCE** is an end-to-end unsupervised scientific intelligence platform that systematically identifies, evaluates, explains, and visualizes anomalous meteorological observations from the **NOAA Global Historical Climatology Network - Daily (GHCN-Daily)** archive. Integrating two complementary unsupervised anomaly detection paradigms—**Isolation Forest** (tree-based recursive partitioning) and **Local Outlier Factor (LOF)** (local reachability density estimation)—the system synthesizes consensus flags, conducts multi-dimensional **Principal Component Analysis (PCA)** projections, ranks suspicious candidates via an uncalibrated composite **Severity Index**, and audits historical extreme weather benchmarks.

---

## 4. Problem Being Addressed
Automated quality assurance of climatological surface observations faces three fundamental scientific and computational challenges:
1. **Intractability of Manual Auditing:** Modern observation archives ingest tens of thousands of measurements daily across vast geographical networks, rendering manual inspection impossible.
2. **The "Outlier vs. Extreme Weather" Dilemma:** An unusually high temperature or rainfall spike may represent a severe sensor short-circuit or a once-in-a-century climate phenomenon. Standard threshold rules struggle to separate sensor failures from natural disasters.
3. **Absence of Ground-Truth Hardware Failure Labels:** Ground meteorological datasets rarely feature verified sensor diagnostic labels. Quality control must therefore rely on unsupervised multivariate learning, multi-model consensus, and domain-informed investigative workflows.

---

## 5. Objectives
- **Objective 1 — Data Ingestion & Physical Validation:** Acquire, decode, and parse 8 full years (2016–2023) of official NOAA GHCN-Daily records for 4 distinct climatic stations ($N = 11,688$ total daily observations), validating thermodynamic consistency (e.g., $T_{\text{max}} \ge T_{\text{min}}$) and official NOAA QC flags.
- **Objective 2 — Robust Feature Engineering:** Construct diurnal thermal ranges, 24-hour day-to-day first differences, and trigonometric cyclical day-of-year encodings ($\sin/\cos$) while preserving heavy distribution tails via z-score `StandardScaler`.
- **Objective 3 — Isolation Forest Deployment:** Implement recursive tree isolation ($T=100$, contamination $\alpha=0.03$), isolating global multivariable outliers far from the distribution centroid.
- **Objective 4 — Local Outlier Factor Deployment:** Implement $k$-nearest neighbors local density estimation ($k=20$, contamination $\alpha=0.03$), isolating localized deviations within tight regional climate clusters.
- **Objective 5 — Model Comparison & The Jaccard Paradox:** Quantify agreement and divergence between tree isolation and density estimation using Jaccard similarity indices, confusion matrices, and station/seasonal breakdown.
- **Objective 6 — Principal Component Analysis (PCA):** Project the 10-dimensional feature space onto 2D and 3D orthogonal coordinate planes to evaluate cluster separation, scree elbow curves, and physical eigenvector loadings.
- **Objective 7 — Multi-Tier Anomaly Investigation:** Implement an uncalibrated composite **Severity Index** ranking heuristic to prioritize high-confidence consensus anomalies ($N = 78$) for human-in-the-loop audit.
- **Objective 8 — Scientific Benchmark Validation:** Cross-reference flagged outliers against documented historical extreme events (2019 Polar Vortex, 2021 PNW Heat Dome, 2021 Hurricane Ida, 2021 Chicago Blizzard) to empirically analyze true climate extremes versus potential sensor defects.
- **Objective 9 — Interactive Intelligence Platform:** Deliver an accessible, responsive, commercial-grade web platform (Flask + Jinja2 + Tailwind CSS + Plotly.js + Leaflet.js) adhering to WCAG 2.1 AA standards.

---

## 6. Dataset Source and Description
The dataset is drawn directly from the **NOAA Global Historical Climatology Network - Daily (GHCN-Daily)** repository managed by the National Centers for Environmental Information (NCEI):
- **Custodian:** NOAA / National Centers for Environmental Information (NCEI)
- **Official Documentation:** [NOAA GHCN-Daily Readme](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt)
- **Temporal Scope:** January 1, 2016 through December 31, 2023 (8 full calendar years, exactly 2,922 days per station including leap years 2016 and 2020).
- **Total Records:** Exactly 11,688 daily multi-sensor observation records.
- **Missing Value Handling:** Missing records encoded as `-9999` in raw data are converted to `NaN` and imputed per-station using forward-fill followed by backward-fill, ensuring continuity without lookahead leakage.

---

## 7. Selected Weather Stations
Four long-standing First-Order National Weather Service stations were selected representing diverse Köppen climate regimes across the United States:

| Station ID | Station Name | Latitude | Longitude | Elevation | Köppen Climate Classification | Records |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **USW00094846** | Chicago O'Hare Int'l Airport, IL | $41.9602^\circ\text{N}$ | $-87.9316^\circ\text{W}$ | $204.8\,\text{m}$ | *Dfa* — Humid Continental (Severe winters, hot summers) | 2,922 |
| **USW00094728** | New York Central Park, NY | $40.7789^\circ\text{N}$ | $-73.9692^\circ\text{W}$ | $42.7\,\text{m}$ | *Cfa / Dfa* — Humid Subtropical / Continental Maritime | 2,922 |
| **USW00024233** | Seattle-Tacoma Int'l Airport, WA | $47.4447^\circ\text{N}$ | $-122.3144^\circ\text{W}$ | $113.1\,\text{m}$ | *Csb* — Warm-Summer Mediterranean / Maritime Oceanic | 2,922 |
| **USW00012839** | Miami Int'l Airport, FL | $25.7906^\circ\text{N}$ | $-80.3164^\circ\text{W}$ | $2.1\,\text{m}$ | *Aw* — Tropical Wet-and-Dry / Savanna | 2,922 |
| **Total** | **4 Distinct Climates** | — | — | — | **National Climatological Cross-Section** | **11,688** |

---

## 8. System Architecture
The application follows an enterprise-grade service-oriented Flask architectural pattern:

```
+-----------------------------------------------------------------------------------------+
|                                    CLIENT BROWSER                                       |
|  Tailwind CSS Design System | Vanilla ES6+ JS | Plotly.js Scientific Charts | Leaflet.js |
+-----------------------------------------------------------------------------------------+
                                           | HTTP Requests (REST JSON & Server HTML)
                                           v
+-----------------------------------------------------------------------------------------+
|                                     FLASK SERVER                                        |
|  +-------------------------------------+  +------------------------------------------+  |
|  |           VIEW BLUEPRINT            |  |              API BLUEPRINT               |  |
|  | 12 Routes (/overview, /eda, etc.)   |  | 50 Endpoints (/api/dataset, /api/models) |  |
|  +-------------------------------------+  +------------------------------------------+  |
+-----------------------------------------------------------------------------------------+
                                           | Internal Service Layer Calls
                                           v
+-----------------------------------------------------------------------------------------+
|                             DATA SERVICE & ML SERVICE LAYER                             |
|  - DataService: In-memory caching, indexing, filtering, pagination, CSV exports        |
|  - MLService: Parameter sensitivity runs, observation vector explainers, triage engine  |
+-----------------------------------------------------------------------------------------+
                                           | Clean Data & ML Artifacts
                                           v
+-----------------------------------------------------------------------------------------+
|                          PERSISTENT DATA & ANALYTICAL ARTIFACTS                         |
|  - data/processed/ghcnd_processed.csv (11,688 rows, 49 columns)                         |
|  - outputs/anomaly_results/consensus_anomaly_results.csv (Enriched with predictions)    |
|  - outputs/comparison_tables/pca_component_loadings.csv & model_comparison_matrix.csv   |
|  - outputs/evaluation_results/ (pca_summary.json, historical_extreme_events_audit.csv)  |
+-----------------------------------------------------------------------------------------+
```

---

## 9. Technology Stack
- **Backend Framework:** Python 3.11.15, Flask 3.1.3, Jinja2 3.1.6, Werkzeug 3.1.9
- **Scientific Computing & ML:** NumPy 2.0.2, Pandas 2.2.3, SciPy 1.14.1, Scikit-learn 1.5.2
- **Data Visualization & Plotting:** Plotly.js 2.29.1, Leaflet.js 1.9.4, Matplotlib 3.9.4, Seaborn 0.13.2
- **Styling & Icons:** Tailwind CSS (via local design tokens), Lucide Icons
- **Automated Testing:** Pytest 8.2.2 (89 unit and integration tests)
- **Academic Deliverables:** fpdf2 2.9.0 (PDF Report Generation), python-pptx 1.0.2 (PowerPoint Generation)

---

## 10. Data Preprocessing
1. **NOAA Unit Decoding:** Raw GHCN-Daily temperatures are stored in tenths of degrees Celsius ($0.1^\circ\text{C}$) and precipitation/snow in tenths of millimeters ($0.1\,\text{mm}$). Preprocessing decodes all variables to SI meteorological standards ($^\circ\text{C}$ and $\text{mm}$).
2. **Missing-Value Elimination:** All `-9999` sentinel values are converted to standard missing values and imputed per station using forward-fill followed by backward-fill, eliminating gaps without seasonal leakage.
3. **Physical Law Consistency Checks:** Every observation is validated against thermodynamic principles:
   $$\text{Rule 1:} \quad T_{\text{max}} \ge T_{\text{min}}$$
   $$\text{Rule 2:} \quad \text{PRCP} \ge 0, \quad \text{SNOW} \ge 0, \quad \text{SNWD} \ge 0$$
   $$\text{Rule 3:} \quad \text{If } T_{\text{min}} > 5.0^\circ\text{C} \implies \text{SNOW} = 0 \text{ (Liquid precipitation only)}$$
4. **Standard Scaling:** Features are standardized using `StandardScaler`:
   $$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
   Standard scaling preserves extreme distribution tails in multi-dimensional space, unlike Min-Max scaling which compresses critical anomaly variance.

---

## 11. Feature Engineering
The pipeline maps each daily record to a 10-dimensional numerical feature space:
$$x_i = [TMAX, TMIN, TEMP\_RANGE, PRCP, SNOW, SNWD, TMAX\_DIFF\_1D, TMIN\_DIFF_1D, SIN\_DOY, COS\_DOY]^T \in \mathbb{R}^{10}$$

1. **Diurnal Temperature Range (`TEMP_RANGE`):**
   $$\text{TEMP\_RANGE}_i = TMAX_i - TMIN_i$$
   Measures daily radiative solar forcing and cloud cover dynamics.
2. **24-Hour Synoptic Delta (`TMAX_DIFF_1D`, `TMIN_DIFF_1D`):**
   $$\Delta TMAX_i = TMAX_i - TMAX_{i-1}, \quad \Delta TMIN_i = TMIN_i - TMIN_{i-1}$$
   Captures abrupt cold/warm front air mass transitions.
3. **Cyclical Day-of-Year Encoding (`SIN_DOY`, `COS_DOY`):**
   $$\text{SIN\_DOY}_i = \sin\left(\frac{2\pi \cdot \text{DOY}_i}{365.25}\right), \quad \text{COS\_DOY}_i = \cos\left(\frac{2\pi \cdot \text{DOY}_i}{365.25}\right)$$
   Preserves continuous seasonal circularity across December 31 to January 1 transitions.

---

## 12. Isolation Forest Explanation
Isolation Forest (Liu, Ting & Zhou, 2008) operates on the principle that anomalous observations are "few and different", rendering them susceptible to rapid isolation under recursive random axis-aligned partitioning.

### Mathematical Formulation
Given an ensemble of $T = 100$ isolation trees, each instance $x$ is recursively partitioned until isolated at leaf depth $h(x)$. The expected path length is normalized by average search depth in a binary search tree of size $n$:
$$c(n) = 2 \left( \ln(n - 1) + 0.5772156649 \right) - \frac{2(n - 1)}{n}$$
The anomaly score $s(x, n)$ is:
$$s(x, n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}$$
Scikit-learn's `decision_function` returns positive values for normal points and negative values for anomalies.
- **Baseline Parameters:** `n_estimators = 100`, `contamination = 0.03`, `random_state = 42`.
- **Anomalies Flagged:** Exactly 351 observations (3.00%).

---

## 13. Local Outlier Factor (LOF) Explanation
Local Outlier Factor (Breunig et al., 2000) measures local density deviation of an observation relative to its $k$-nearest neighbors, effectively identifying outliers in non-uniform or clustered density spaces.

### Mathematical Formulation
1. **$k$-distance neighborhood $N_k(p)$:** The set of points within Euclidean distance $d_k(p)$.
2. **Reachability Distance:**
   $$\text{reach-dist}_k(p, o) = \max\{d_k(o), d(p, o)\}$$
3. **Local Reachability Density ($lrd_k$):**
   $$\text{lrd}_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)}$$
4. **Local Outlier Factor ($LOF_k$):**
   $$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}}{|N_k(p)|}$$
- **Baseline Parameters:** `n_neighbors = 20`, `contamination = 0.03`, `novelty = False`.
- **Anomalies Flagged:** Exactly 351 observations (3.00%).

---

## 14. Model Comparison Methodology & The Jaccard Paradox
Comparing Isolation Forest and LOF across the 11,688 observations reveals a critical machine learning insight termed **The Jaccard Paradox**:

| Metric | Recalculated Value | Explanation |
| :--- | :---: | :--- |
| **Total Observations** | 11,688 | 4 stations, 8 full years (2016–2023) |
| **Isolation Forest Flagged** | 351 (3.00%) | Global multivariable tail outliers |
| **Local Outlier Factor Flagged** | 351 (3.00%) | Local density-degraded outliers |
| **Consensus Intersection ($IF \cap LOF$)** | **78 (0.67%)** | High-confidence outliers flagged by both algorithms |
| **IF-Only Outliers ($IF \setminus LOF$)** | 273 (2.34%) | Flagged exclusively by Isolation Forest |
| **LOF-Only Outliers ($LOF \setminus IF$)** | 273 (2.34%) | Flagged exclusively by Local Outlier Factor |
| **Union of Outliers ($IF \cup LOF$)** | 624 (5.34%) | Observations flagged by at least one model |
| **Concordant Inliers** | 11,064 (94.66%) | Observations considered normal by both models |
| **Total Binary Agreement** | **95.33%** | $(11,064 + 78) / 11,688$ concordance |
| **Jaccard Similarity Index** | **0.1250 (12.50%)** | $|IF \cap LOF| / |IF \cup LOF| = 78 / 624$ |

### Scientific Significance of the Paradox
While overall binary agreement is high ($95.33\%$), this agreement is almost entirely driven by common normal inliers. On the critical $3\%$ outlier tails, the models disagree on $7$ out of every $8$ detections ($J = 0.1250$). This proves that Isolation Forest and LOF capture fundamentally different geometric failure modes: Isolation Forest isolates global extremes situated far from the multidimensional center, whereas LOF isolates local density drops relative to neighbors within equable regional clusters.

---

## 15. Principal Component Analysis (PCA) Methodology
PCA transforms the standardized 10-dimensional meteorological feature space onto orthogonal axes:

### Authoritative PCA Specification:
- **Input Features ($P = 10$):** `['TMAX', 'TMIN', 'TEMP_RANGE', 'PRCP', 'SNOW', 'SNWD', 'TMAX_DIFF_1D', 'TMIN_DIFF_1D', 'SIN_DOY', 'COS_DOY']`
- **Observations Fitted:** Exactly 11,688.
- **Fitting Method:** Scikit-learn `PCA(n_components=10, random_state=42)`.
- **Explained Variance Breakdown:**
  - **PC1:** $27.96\%$ (Thermal & Annual Cycle Axis)
  - **PC2:** $16.08\%$ (Synoptic Volatility & Diurnal Range Axis)
  - **PC3:** $12.20\%$ (Precipitation & Convective Dynamics Axis)
  - **PC4:** $11.45\%$ | **PC5:** $9.52\%$ | **PC6:** $8.70\%$ | **PC7:** $7.01\%$ | **PC8:** $4.24\%$ | **PC9:** $2.83\%$
  - **PC10:** **$0.00\%$** (Collinear Null Subspace)
- **Cumulative Variance:**
  - **2D Cumulative (PC1 + PC2):** **$44.05\%$**
  - **3D Cumulative (PC1 + PC2 + PC3):** **$56.25\%$**

### Mathematical Derivation of Rank-9 Subspace (PC10 = 0.00%)
Because `TEMP_RANGE` is algebraically constructed as $\text{TEMP\_RANGE} = \text{TMAX} - \text{TMIN}$, the columns of the unstandardized data matrix are linearly dependent. Standard scaling applies column-specific affine transformations ($z = (x - \mu)/\sigma$), which maps this relationship into an exact linear combination in the standardized space:
$$z_{\text{TEMP\_RANGE}} = \frac{\sigma_{\text{TMAX}}}{\sigma_{\text{TEMP\_RANGE}}} z_{\text{TMAX}} - \frac{\sigma_{\text{TMIN}}}{\sigma_{\text{TEMP\_RANGE}}} z_{\text{TMIN}} + C$$
Consequently, the covariance matrix has rank exactly $9$, meaning the 10th eigenvalue is mathematically $\lambda_{10} = 0$, explaining $0.00\%$ of total variance.

### Eigenvector Feature Loadings:
| Feature | PC1 Loading | PC2 Loading | PC3 Loading | Primary Physical Driver |
| :--- | :---: | :---: | :---: | :--- |
| `TMAX` | **+0.5715** | -0.0405 | +0.0487 | Maximum ambient air temperature |
| `TMIN` | **+0.5333** | -0.1861 | +0.2021 | Minimum ambient air temperature |
| `COS_DOY` | **-0.4394** | +0.0498 | +0.1209 | Winter polar chill vs. summer solar heating |
| `SNWD` | -0.2453 | +0.1790 | +0.2265 | Surface snow pack accumulation depth |
| `TEMP_RANGE` | +0.2092 | **+0.4321** | **-0.4540** | Daily diurnal radiative temperature range |
| `SNOW` | -0.1814 | +0.0263 | +0.3357 | 24-hour snowfall accumulation |
| `SIN_DOY` | -0.1574 | +0.2424 | -0.2318 | Spring-to-autumn seasonal transition |
| `TMAX_DIFF_1D` | +0.1471 | **+0.6624** | +0.1573 | 24-hour maximum temperature delta |
| `TMIN_DIFF_1D` | +0.1098 | **+0.4437** | **+0.5265** | 24-hour minimum temperature delta |
| `PRCP` | +0.0248 | -0.2176 | **+0.4652** | 24-hour liquid precipitation accumulation |

---

## 16. Anomaly Investigation Methodology & Severity Index
To assist human meteorologists in prioritizing observations for diagnostic audit, the platform calculates an uncalibrated composite **Severity Index**:

$$\text{SEVERITY\_INDEX} = (-\text{IF\_SCORE}) \times \left((-\text{LOF\_SCORE}) - 1.0\right)$$

### Sign Direction & Terms:
1. **Isolation Forest Term ($-\text{IF\_SCORE}$):** In scikit-learn, nominal points receive positive decision scores, while anomalies receive negative scores. Multiplying by $-1$ ensures that higher positive values ($> 0$) reflect deeper tree isolation.
2. **Local Outlier Factor Term ($(-\text{LOF\_SCORE}) - 1.0$):** Scikit-learn's `negative_outlier_factor_` is $-1.0$ for nominal points in uniform density and $<-1.0$ for outliers. Multiplying by $-1$ and subtracting $1.0$ isolates the excess local reachability distance over nominal density ($\text{LOF} - 1.0 \ge 0$).
3. **Compound Ranking:** The product compounds the two factors, elevating points that are both globally isolated in tree space and locally sparse compared to neighbors.
4. **Mandatory Scientific Disclaimer:** The Severity Index is an uncalibrated ranking heuristic for triage prioritization. It does not represent a calibrated probability of sensor hardware failure.

---

## 17. Scientific Limitations
1. **Uncalibrated Attribution:** In the absence of physical sensor maintenance logs, unsupervised learning isolates statistical anomalies, not confirmed hardware defects.
2. **Climatic Masking:** Pooled national models exhibit geographical bias: continental stations with severe seasonal swings (Chicago: 7.77% anomaly rate) trigger more flags than equable tropical stations (Miami: 1.06% anomaly rate).
3. **Extreme Weather Confounding:** Genuine meteorological records (e.g., hurricane flash floods, polar vortexes) look mathematically identical to sensor spikes under distance-based metrics.
4. **Information Discard in 2D PCA:** 2D PCA visual projections capture $44.05\%$ of feature variance; $55.95\%$ of multi-dimensional variance is discarded in 2D scatter plots.

---

## 18. Ethical Considerations
- **No False Alarms for Public Safety:** Weather anomaly models must not be allowed to automatically delete or censor observations from weather forecasting models without human verification, as suppressing extreme weather signals could delay life-saving evacuation warnings.
- **Data Integrity & Traceability:** All original NOAA observations, raw values, and quality flags remain intact; no historical record is overwritten or deleted.

---

## 19. Project Directory Structure
```
Faulty_Weather_Station_Detection/
│
├── run.py                                # Application entry point (python run.py)
├── requirements.txt                      # Verified Python dependencies
├── .gitignore                            # Clean Git exclusion rules
├── PROJECT_LINKS.txt                     # Submission metadata, links & 9-step demo sequence
├── README.md                             # Comprehensive project documentation
│
├── app/                                  # Flask Application Package
│   ├── __init__.py                       # Application factory (create_app())
│   ├── config.py                         # Application configuration
│   ├── routes/
│   │   ├── __init__.py                   # Route blueprint registration
│   │   ├── views.py                      # 12 View routes (/overview, /eda, etc.)
│   │   └── api.py                        # 50 JSON API endpoints (/api/*)
│   ├── services/
│   │   ├── __init__.py                   # Service exports
│   │   ├── data_service.py               # In-memory caching, filtering, PCA, and triage
│   │   └── ml_service.py                 # Dynamic model evaluation and sensitivity runs
│   ├── static/                           # Static assets
│   │   ├── css/atmosphere.css            # Custom CSS design tokens, WCAG focus-visible
│   │   └── js/                           # Vanilla ES6+ modules
│   │       ├── core.js                   # Universal drawer, Plotly resize observer, search
│   │       ├── dataset_explorer.js       # Live Leaflet map, table pagination, filter
│   │       ├── eda.js                    # EDA distributions, correlation heatmap
│   │       ├── data_quality.js           # NOAA QC flags, feature distributions
│   │       ├── isolation_forest.js       # IF sensitivity charts and distributions
│   │       ├── lof.js                    # LOF sensitivity and score curves
│   │       ├── model_comparison.js       # Jaccard Venn, agreement matrices, breakdown
│   │       ├── pca_analysis.js           # Interactive 2D/3D PCA scatter and Scree plot
│   │       ├── anomaly_investigation.js  # Triage severity table, observation inspector
│   │       └── scientific_insights.js    # Historical benchmark audits, bias charts
│   └── templates/                        # Jinja2 HTML templates
│       ├── base.html                     # Master layout (sidebar, topbar, theme switcher)
│       ├── components/                   # Reusable components (sidebar, topbar, stat card)
│       ├── overview.html                 # Module 1: Project Overview
│       ├── dataset_explorer.html         # Module 2: Dataset Explorer
│       ├── eda.html                      # Module 3: Exploratory Data Analysis
│       ├── data_quality.html             # Module 4: Data Quality & Preprocessing
│       ├── isolation_forest.html         # Module 5: Isolation Forest
│       ├── local_outlier_factor.html     # Module 6: Local Outlier Factor
│       ├── model_comparison.html         # Module 7: Model Comparison Engine
│       ├── pca_visualization.html        # Module 8: PCA 2D/3D Dimensionality Reduction
│       ├── anomaly_investigation.html    # Module 9: Anomaly Investigation
│       └── scientific_insights.html      # Module 10: Scientific Insights
│
├── data/                                 # Meteorological Data
│   ├── raw/                              # Original raw NOAA GHCN-Daily CSV files
│   │   ├── USW00094846.csv               # Chicago O'Hare
│   │   ├── USW00094728.csv               # New York Central Park
│   │   ├── USW00024233.csv               # Seattle-Tacoma
│   │   └── USW00012839.csv               # Miami
│   └── processed/
│       └── ghcnd_processed.csv           # Cleaned, imputed, scaled dataset (11,688 rows)
│
├── src/                                  # Modular Python Core Library
│   ├── __init__.py
│   ├── data_loader.py                    # NOAA dataset ingestion and parsing
│   ├── data_preprocessing.py             # Feature engineering, scaling, physical checks
│   ├── models.py                         # Scikit-learn Isolation Forest & LOF wrappers
│   ├── detection.py                      # Baseline experiment orchestration
│   ├── pca_analysis.py                   # 10D PCA fitting, loadings, and publication graphs
│   └── pipeline.py                       # End-to-end execution script
│
├── outputs/                              # Analytical Outputs & Precomputed Results
│   ├── anomaly_results/
│   │   ├── consensus_anomaly_results.csv # Master dataset with all predictions and PCs
│   │   ├── isolation_forest_results.csv  # IF anomaly scores and flags
│   │   └── lof_results.csv               # LOF negative outlier factor scores and flags
│   ├── comparison_tables/
│   │   ├── model_comparison_matrix.csv   # Confusion matrix and agreement metrics
│   │   └── pca_component_loadings.csv    # Eigenvector feature loadings
│   └── evaluation_results/
│       ├── comparative_summary.json      # Key comparative metrics
│       ├── geographical_station_bias.csv # Anomaly rates across stations
│       ├── historical_extreme_events_audit.csv # Historical event validation
│       ├── isolation_forest_summary.json # IF parameter metadata
│       ├── lof_summary.json              # LOF parameter metadata
│       ├── pca_summary.json              # Variance ratios and top loadings
│       ├── preprocessing_summary.json    # Preprocessing audit
│       ├── seasonal_bias.csv             # Seasonal anomaly breakdown
│       └── top_suspicious_observations.csv # Top Severity Index candidates
│
├── graphs/                               # Publication-Quality Generated Figures
│   └── pca/                              # Matplotlib/Seaborn 300 DPI PCA plots
│
├── reports/                              # Project Reports & Screenshots
│   ├── screenshots/phase_i/              # 30 responsive verification screenshots
│   ├── ATMOSPHERE_INTELLIGENCE_FINAL_REPORT.pdf  # Final Academic Report (PDF)
│   └── ATMOSPHERE_INTELLIGENCE_FINAL_PRESENTATION.pptx # Final Presentation (PPTX)
│
├── docs/                                 # Technical Documentation
│   ├── architecture.md                   # System architecture documentation
│   └── methodology.md                    # Detailed mathematical methodology
│
└── tests/                                # Automated Pytest Suite (89 tests)
    ├── test_environment.py               # Dependencies and environment tests
    ├── test_data.py                      # Data ingestion and NOAA parsing tests
    ├── test_preprocessing.py             # Feature engineering and physical rules
    ├── test_detection.py                 # Core ML models and PCA tests
    ├── test_layout.py                    # Flask layout, views, static assets
    ├── test_dataset_explorer.py          # Module 1 & 2 view and API tests
    ├── test_eda_quality.py               # Module 3 & 4 view and API tests
    ├── test_models.py                    # Module 5 & 6 view and API tests
    ├── test_comparison_pca.py            # Module 7 & 8 view and API tests
    └── test_investigation_insights.py    # Module 9 & 10 view and API tests
```

---

## 20. Installation Instructions
Ensure Python 3.11+ is installed on your workstation.

```bash
# 1. Clone or unpack the project
cd Faulty_Weather_Station_Detection

# 2. Create a clean virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 4. Install verified dependencies
pip install -r requirements.txt
```

---

## 21. Environment Setup
No external API keys, database credentials, or third-party cloud accounts are required. All processing runs entirely on local compute using offline NOAA GHCN-Daily datasets.
- Ensure the `MPLCONFIGDIR` environment variable points to a writable local directory to avoid permission locks on restricted systems:
  ```powershell
  $env:MPLCONFIGDIR = "$PWD\.cache\matplotlib"
  ```

---

## 22. How to Run the Application
To start the ATMOSPHERE INTELLIGENCE web platform:
```bash
python run.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```
The platform automatically loads precomputed analytical models and provides real-time access to all 10 modules.

---

## 23. How to Run Tests
Execute the comprehensive Pytest verification suite:
```bash
pytest tests/ -v
```
Expected output:
```text
====================== 89 passed, 14 warnings in 22.64s =======================
```
All 89 tests will pass, validating data loaders, physical constraints, model determinism, PCA projections, API schemas, and view rendering.

---

## 24. API Overview
ATMOSPHERE INTELLIGENCE exposes 50 REST JSON endpoints under `/api/*`:
- `/api/health`: Health status, observation count, station list.
- `/api/overview/metrics`: High-level KPI summary cards.
- `/api/dataset/summary`: Feature types, records, missing values.
- `/api/dataset/records`: Paginated, filtered observation rows (supports `page`, `per_page`, `station`, `status`, `search`).
- `/api/dataset/export-csv`: Generates filtered CSV downloads.
- `/api/eda/distributions/<feature>`: Empirical histograms, KDEs, summary statistics.
- `/api/eda/correlations`: Pearson correlation matrix.
- `/api/models/isolation-forest/results`: Baseline Isolation Forest predictions and score distributions.
- `/api/models/isolation-forest/sensitivity`: Contamination parameter sensitivity comparisons ($\alpha \in \{0.01, 0.03, 0.05\}$).
- `/api/models/lof/results`: Baseline LOF predictions and score distributions.
- `/api/models/lof/sensitivity`: Neighborhood sensitivity comparisons ($k \in \{10, 20, 30\}$).
- `/api/comparison/summary`: Agreement metrics, confusion matrix, Jaccard similarity.
- `/api/pca/summary`: Explained variance ratios, scree plot points, top eigenvector loadings.
- `/api/pca/coordinates`: 2D/3D projected coordinates with anomaly classifications.
- `/api/investigation/observations`: Paginated anomaly triage table sorted by Severity Index.
- `/api/investigation/observation/<id>`: Full feature vector and 5-nearest neighbors explainer.
- `/api/insights/extreme-events`: Historical benchmark validation audit.
- `/api/insights/station-bias` & `/api/insights/seasonal-bias`: Geographic and seasonal skew analytics.

---

## 25. Screenshot Gallery
30 responsive verification screenshots are permanently available in [reports/screenshots/phase_i/](file:///c:/Users/kalis/.gemini/antigravity-ide/scratch/New%20Project/Faulty_Weather_Station_Detection/reports/screenshots/phase_i/):
- **Desktop ($1440\,\text{px}$):** Complete views of all 10 modules.
- **Tablet ($768\,\text{px}$):** Portrait and landscape reflow verification.
- **Mobile ($390\,\text{px}$):** Collapsible drawer, single-column KPI stacks, and touch-scroll tables.

---

## 26. Reproducibility Instructions
To re-run the entire data processing and ML pipeline from raw NOAA data from scratch:
```bash
python src/pipeline.py --full
```
This command:
1. Reloads raw station CSVs from `data/raw/`.
2. Decodes NOAA units, applies forward/backward-fill imputation, and audits physical laws.
3. Computes engineered features (`TEMP_RANGE`, deltas, `SIN_DOY`, `COS_DOY`).
4. Re-fits Isolation Forest (`random_state=42`) and LOF.
5. Re-fits 10D PCA (`random_state=42`) and outputs loadings and summary JSONs.
6. Generates high-resolution publication charts in `graphs/pca/`.

All pipeline steps are deterministic and guaranteed to produce identical metrics.

---

## 27. Known Limitations
- **No Ground-Truth Sensor Flags:** Findings represent statistical outliers, not confirmed physical sensor hardware failures.
- **Climatological Regional Skew:** Pooled models disproportionately flag continental stations experiencing extreme natural weather.
- **Dimensionality Reduction Information Loss:** 2D PCA plots explain $44.05\%$ of feature variance and discard $55.95\%$.

---

## 28. Credits and References
- **Dataset:** NOAA National Centers for Environmental Information (NCEI) GHCN-Daily archive.
- **Isolation Forest:** Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. Eighth IEEE International Conference on Data Mining (ICDM), pp. 413-422.
- **Local Outlier Factor:** Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). *LOF: identifying density-based local outliers*. ACM SIGMOD Record, 29(2), pp. 93-104.
- **Principal Component Analysis:** Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer Series in Statistics.
- **Course & Mentorship:** INT396 — Unsupervised Machine Learning, Department of Computer Science & Engineering.
