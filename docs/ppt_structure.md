# University Presentation Structure (15 Slides)

**Course:** INT396 – Unsupervised Machine Learning  
**Problem Statement ID:** SIH26073  
**Project Title:** Faulty Weather-Station Observation Detection Using Anomaly Detection Methods  

---

### Slide 1: Title Slide
- **Title:** Faulty Weather-Station Observation Detection
- **Subtitle:** Unsupervised Anomaly Detection using Isolation Forest, LOF & PCA (SIH26073)
- **Course:** INT396 – Unsupervised Machine Learning
- **Presenters:** [PENDING: Student Names & Registration Numbers]
- **Mentor:** [PENDING: Faculty Instructor Name]

### Slide 2: Introduction & Motivation
- Weather stations collect continuous ground observations ($TMAX, TMIN, PRCP, SNOW, SNWD$).
- Sensor malfunctions (freezing, hardware failure, transmission losses) compromise data integrity.
- Manual verification across millions of records is intractable; automated detection is vital.

### Slide 3: Problem Statement (SIH26073)
- Challenge: Detecting anomalous weather readings without ground truth labels.
- Key Scientific Dilemma: Distinguishing genuine extreme weather (polar vortex, heat dome) from actual sensor defects.
- Goal: Implement, compare, and visualize unsupervised anomaly detection algorithms.

### Slide 4: Project Objectives
- Acquire real NOAA GHCN-Daily multi-station data.
- Audit physical rules ($TMIN \le TMAX$, $PRCP \ge 0$) and handle sentinel codes (`-9999`).
- Feature engineer diurnal temperature range and cyclical temporal encodings.
- Train Isolation Forest across contaminations ($0.01, 0.03, 0.05$).
- Train Local Outlier Factor across neighbor sizes ($k=10, 20, 30$).
- Compare models via Jaccard similarity and agreement percentages.
- Project into 2D space with PCA and analyze eigenvector loadings.
- Develop interactive Streamlit web dashboard.

### Slide 5: NOAA GHCN-Daily Dataset Description
- 4 Stations across distinct climate zones:
  - New York Central Park (`USW00094728`): Continental / Maritime
  - Chicago O'Hare (`USW00094846`): Continental / Severe Winters
  - Seattle Tacoma Airport (`USW00024233`): Pacific Oceanic
  - Miami Airport (`USW00012839`): Subtropical Savanna
- Period: 2016 to 2023 (8 full years = 2,922 days/station; 11,688 total observations).
- Decoded units: tenths of $^\circ\text{C} \to ^\circ\text{C}$, tenths of $\text{mm} \to \text{mm}$.

### Slide 6: System Architecture & Workflow
- Flow: Raw NOAA Ingestion $\to$ Physical QC Audit $\to$ Feature Engineering $\to$ Scaling $\to$ Dual Unsupervised Models $\to$ Comparison Engine $\to$ 2D PCA Projection $\to$ Streamlit UI.
- Preserves raw data separately from processed data.

### Slide 7: Preprocessing & Feature Engineering
- Sentinel handling: `-9999 \to NaN`.
- Imputation: Forward-fill + station seasonal median.
- Engineered Features:
  - $TEMP\_RANGE = TMAX - TMIN$
  - $\sin(2\pi \cdot DOY / 365.25), \cos(2\pi \cdot DOY / 365.25)$
  - 1-Day Temporal Differences: $\Delta TMAX, \Delta TMIN$
- Feature Scaling: `StandardScaler` (zero mean, unit variance).

### Slide 8: Isolation Forest Anomaly Detection
- Theory: Isolates anomalies using recursive random axis-aligned splits; anomalies have shorter path lengths $h(x)$.
- Anomaly score: $s(x, n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}$.
- Experiments:
  - $c=0.01$: 117 anomalies (1.00%)
  - $c=0.03$: 351 anomalies (3.00%) [Baseline]
  - $c=0.05$: 585 anomalies (5.01%)

### Slide 9: Local Outlier Factor (LOF) Detection
- Theory: Measures local density deviation relative to $k$-nearest neighbors using local reachability density ($lrd_k$).
- Flags points situated in sparse regions compared to their local cluster.
- Experiments ($c=0.03$):
  - $k=10$: 351 anomalies (Offset: -1.356)
  - $k=20$: 351 anomalies (Offset: -1.363) [Baseline]
  - $k=30$: 351 anomalies (Offset: -1.377)

### Slide 10: Comparative Analysis & Jaccard Similarity
- Baseline Comparison ($IF_{0.03}$ vs $LOF_{20}$):
  - Consensus Anomalies ($IF \cap LOF$): **78**
  - Isolation Forest Only: **273** (captures global extremes)
  - Local Outlier Factor Only: **273** (captures local density anomalies)
  - Total Agreement: **95.33%** | Jaccard Similarity: **0.1250**
- Sensitivity Grid shows highest stability at $c=0.03, k=20$.

### Slide 11: Principal Component Analysis (PCA) 2D Visualization
- PCA reduces 10D scaled feature space into 2 orthogonal axes.
- **PC1 (27.96% Variance):** Thermal baseline ($TMAX$ +0.572, $TMIN$ +0.533, $\cos(DOY)$ -0.439).
- **PC2 (16.08% Variance):** Dynamic volatility ($\Delta TMAX$ +0.662, $TEMP\_RANGE$ +0.432).
- **Total Explained Variance: 44.05%**.
- Visualizes clear separation of consensus anomalies along distribution perimeters.

### Slide 12: Evaluation & Historical Extreme Events Validation
- Validated on documented real-world climate extremes:
  - **2019 Chicago Polar Vortex (-30.5°C):** Flagged 3/3 days by both IF and LOF.
  - **2021 Hurricane Ida NYC Rainstorm (>180 mm):** Flagged by both models.
  - **2021 Pacific NW Heat Dome (Seattle 42.2°C):** Flagged by LOF.
- **Key Insight:** These are legitimate meteorological records, NOT hardware faults. Demonstrates why unsupervised models are triage tools, not ground truth classifiers.

### Slide 13: Geographical & Seasonal Bias
- Geographical: Chicago (high variance) produced 7.77% IF anomaly rate vs. Miami (equable tropical) with only 0.24%.
- Seasonal: Winter produced 9.66% IF anomaly rate vs. Summer with 0.17%.
- Conclusion: Clustered regional baselines are required to avoid climate bias.

### Slide 14: Individual Group Contributions
- Member 1: NOAA ingestion pipeline, unit decoding, physical QC audits, automated test suite.
- Member 2: Feature engineering, Isolation Forest parameter experiments, PCA projection and loadings.
- Member 3: LOF implementation, comparative analysis engine, Streamlit dashboard, historical event validation.

### Slide 15: Conclusion & Future Scope
- Successfully delivered a modular, reproducible, university-level anomaly detection system.
- Full agreement of 95.33% demonstrates complementary strengths of tree isolation and density estimation.
- Future work: Spatial neighbor cross-validation (< 50 km) and semi-supervised active learning feedback loops.
- References: Liu et al. (2008), Breunig et al. (2000), Menne et al. (2012).

