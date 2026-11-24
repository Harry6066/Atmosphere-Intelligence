"""Generator script for interactive student Jupyter Notebooks.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Creates structured .ipynb notebooks with educational markdown and reproducible code.
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.15"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


# -------------------------------------------------------------
# 01_data_understanding.ipynb
# -------------------------------------------------------------
nb01 = make_notebook([
    md_cell("""# 01. NOAA GHCN-Daily Data Understanding & Ingestion
**Course:** INT396 – Unsupervised Machine Learning  
**Problem Statement ID:** SIH26073  
**Project Title:** Faulty Weather-Station Observation Detection Using Anomaly Detection Methods

### Notebook Objectives:
1. Understand the official NOAA GHCN-Daily file structure and record schema.
2. Ingest weather station records across diverse climatic zones.
3. Inspect core weather features: `TMAX`, `TMIN`, `PRCP`, `SNOW`, `SNWD`.
4. Inspect NOAA quality flags (`QFLAG`) and sentinel missing values (`-9999`)."""),
    code_cell("""import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import acquire_and_build_dataset, DEFAULT_STATIONS
print("Environment and packages loaded successfully.")"""),
    code_cell("""# Ingest multi-station NOAA records (2016-2023)
raw_df = acquire_and_build_dataset(start_year=2016, end_year=2023)
print(f"Total Observations Loaded: {len(raw_df):,}")
raw_df.head()"""),
    code_cell("""# Inspect station metadata breakdown
raw_df.groupby(["STATION", "NAME"]).agg(
    Observations=("DATE", "count"),
    Start_Date=("DATE", "min"),
    End_Date=("DATE", "max")
)"""),
    code_cell("""# Inspect Missing Values and Sentinel Codes (-9999)
print("Missing values count per column:")
print(raw_df[["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]].isnull().sum())
print("\nOccurrences of NOAA -9999 sentinel code:")
print((raw_df[["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]] == -9999).sum())""")
])
with open(NOTEBOOKS_DIR / "01_data_understanding.ipynb", "w") as f:
    json.dump(nb01, f, indent=2)


# -------------------------------------------------------------
# 02_data_preprocessing.ipynb
# -------------------------------------------------------------
nb02 = make_notebook([
    md_cell("""# 02. Data Preprocessing, Physical Audits & Feature Engineering
**Course:** INT396 – Unsupervised Machine Learning  
**Problem Statement ID:** SIH26073

### Notebook Objectives:
1. Decode NOAA units (tenths of degrees Celsius, tenths of millimeters).
2. Handle sentinel missing values and apply non-destructive imputation.
3. Engineer diurnal temperature range (`TEMP_RANGE`), cyclical seasonal encodings, and temporal differences.
4. Scale features using `StandardScaler` without clipping anomaly tails."""),
    code_cell("""import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import preprocess_pipeline, NUMERICAL_FEATURE_COLS
df_processed, df_scaled, scaler = preprocess_pipeline()
print(f"Processed Observations: {len(df_processed):,}")"""),
    code_cell("""# Summary statistics of preprocessed features
df_processed[NUMERICAL_FEATURE_COLS].describe().round(2).T"""),
    code_cell("""# Inspect Physical Consistency Audit (Inverted Temperatures: TMIN > TMAX)
inverted = (df_processed["TMIN"] > df_processed["TMAX"]).sum()
print(f"Count of Inverted Temperature Violations: {inverted}")"""),
    code_cell("""# Visualize Diurnal Temperature Range vs TMAX
plt.figure(figsize=(9, 5))
sns.scatterplot(data=df_processed, x="TMAX", y="TEMP_RANGE", hue="STATION", alpha=0.5, s=20)
plt.axhline(0, color="red", linestyle="--", label="Physical Bound")
plt.title("Diurnal Range vs. Maximum Temperature (°C)")
plt.legend()
plt.show()""")
])
with open(NOTEBOOKS_DIR / "02_data_preprocessing.ipynb", "w") as f:
    json.dump(nb02, f, indent=2)


# -------------------------------------------------------------
# 03_isolation_forest.ipynb
# -------------------------------------------------------------
nb03 = make_notebook([
    md_cell("""# 03. Isolation Forest Anomaly Detection
**Course:** INT396 – Unsupervised Machine Learning  
**Problem Statement ID:** SIH26073

### Algorithm Formulation:
Isolation Forest isolates anomalies by recursively creating random axis-aligned splits.
Because anomalies possess unusual multivariate feature values, they are isolated in fewer splits (shorter path lengths).
- Contamination setting $\alpha \in \{0.01, 0.03, 0.05\}$ controls the decision threshold.
- Decision score: $s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.isolation_forest import run_isolation_forest_experiments
df_if, if_metrics = run_isolation_forest_experiments()
print("Isolation Forest Experiments Completed:")
for k, v in if_metrics.items():
    print(f" - {k}: {v['anomalies_detected']} anomalies ({v['anomaly_percentage']}%)")"""),
    code_cell("""# Plot Anomaly Score Distribution across Contaminations
plt.figure(figsize=(9, 4.5))
for c in [0.01, 0.03, 0.05]:
    col = f"IF_SCORE_C{int(c*100):02d}"
    sns.kdeplot(df_if[col], label=f"c={c}", fill=True, alpha=0.15)
plt.axvline(0, color="red", linestyle="--", label="Score = 0")
plt.title("Isolation Forest Decision Score Distributions")
plt.xlabel("Anomaly Score (Negative = Outlier)")
plt.legend()
plt.show()"""),
    code_cell("""# Scatter Plot of Detected Anomalies (c=0.03)
plt.figure(figsize=(9, 5))
normal = df_if[df_if["IF_ANOMALY"] == 0]
anom = df_if[df_if["IF_ANOMALY"] == 1]
plt.scatter(normal["TMIN"], normal["TMAX"], c="lightgray", alpha=0.3, s=15, label="Normal")
plt.scatter(anom["TMIN"], anom["TMAX"], c="crimson", alpha=0.85, s=30, label="IF Anomaly (c=0.03)")
plt.title("Isolation Forest Anomalies on Temperature Plane")
plt.xlabel("TMIN (°C)")
plt.ylabel("TMAX (°C)")
plt.legend()
plt.show()""")
])
with open(NOTEBOOKS_DIR / "03_isolation_forest.ipynb", "w") as f:
    json.dump(nb03, f, indent=2)


# -------------------------------------------------------------
# 04_local_outlier_factor.ipynb
# -------------------------------------------------------------
nb04 = make_notebook([
    md_cell("""# 04. Local Outlier Factor (LOF) Anomaly Detection
**Course:** INT396 – Unsupervised Machine Learning  
**Problem Statement ID:** SIH26073

### Algorithm Formulation:
Local Outlier Factor (LOF) measures the local density deviation of an observation with respect to its $k$-nearest neighbors.
- Reachability distance: $reach\\text{-}dist_k(p, o) = \\max\\{d_k(o), d(p, o)\\}$
- Local reachability density: $lrd_k(p) = \\frac{|N_k(p)|}{\\sum_{o \\in N_k(p)} reach\\text{-}dist_k(p, o)}$
- Outlier factor: $LOF_k(p) = \\frac{\\sum_{o \\in N_k(p)} \\frac{lrd_k(o)}{lrd_k(p)}}{|N_k(p)|}$
Observations with substantially lower local density than their neighborhood are flagged as outliers."""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lof_detection import run_lof_experiments
df_lof, lof_metrics = run_lof_experiments()
print("LOF Experiments Completed:")
for k, v in lof_metrics.items():
    print(f" - {k}: {v['anomalies_detected']} anomalies ({v['anomaly_percentage']}%) | Offset: {v['offset_threshold']:.3f}")"""),
    code_cell("""# Negative Outlier Score Distributions
plt.figure(figsize=(9, 4.5))
for k in [10, 20, 30]:
    sns.kdeplot(df_lof[f"LOF_SCORE_K{k}"], label=f"k={k}", fill=True, alpha=0.15)
plt.title("LOF Negative Outlier Score Distribution across k")
plt.xlabel("Negative Outlier Factor")
plt.legend()
plt.show()"""),
    code_cell("""# Scatter Plot of Detected Anomalies (k=20)
plt.figure(figsize=(9, 5))
normal = df_lof[df_lof["LOF_ANOMALY"] == 0]
anom = df_lof[df_lof["LOF_ANOMALY"] == 1]
plt.scatter(normal["TMIN"], normal["TMAX"], c="lightgray", alpha=0.3, s=15, label="Normal")
plt.scatter(anom["TMIN"], anom["TMAX"], c="royalblue", alpha=0.85, s=30, label="LOF Anomaly (k=20)")
plt.title("Local Outlier Factor Anomalies on Temperature Plane")
plt.xlabel("TMIN (°C)")
plt.ylabel("TMAX (°C)")
plt.legend()
plt.show()""")
])
with open(NOTEBOOKS_DIR / "04_local_outlier_factor.ipynb", "w") as f:
    json.dump(nb04, f, indent=2)


# -------------------------------------------------------------
# 05_pca_visualization.ipynb
# -------------------------------------------------------------
nb05 = make_notebook([
    md_cell("""# 05. Principal Component Analysis (PCA) 2D Visualization
**Course:** INT396 – Unsupervised Machine Learning  
**Problem Statement ID:** SIH26073

### Theoretical Concept:
Principal Component Analysis (PCA) is an orthogonal linear transformation that projects high-dimensional data into uncorrelated principal components.
PCA serves as a **dimensionality reduction and visualization technique** to inspect anomaly clustering in 2D space, NOT as the detection algorithm itself."""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pca_analysis import run_pca_pipeline, compute_component_loadings
df_pca, pca_model, var_metrics = run_pca_pipeline()

ev1 = var_metrics['pc1_explained_variance'] * 100
ev2 = var_metrics['pc2_explained_variance'] * 100
print(f"PC1 Explained Variance: {ev1:.2f}%")
print(f"PC2 Explained Variance: {ev2:.2f}%")
print(f"Total Cumulative (2D): {ev1+ev2:.2f}%")"""),
    code_cell("""# Inspect Component Loadings (Eigenvectors)
from src.data_preprocessing import NUMERICAL_FEATURE_COLS
loadings = compute_component_loadings(pca_model, NUMERICAL_FEATURE_COLS)
loadings[["PC1_Loading", "PC2_Loading"]].round(3)"""),
    code_cell("""# 2D PCA Scatter: Consensus vs Method-Specific Anomalies
plt.figure(figsize=(9, 6.5))
palette = {"Normal": "#bdc3c7", "Consensus Anomaly": "#e74c3c", "IF Only": "#e67e22", "LOF Only": "#2980b9"}
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="CONSENSUS_LABEL", palette=palette, alpha=0.75, s=25)
plt.title(f"2D PCA Projection of Weather Observations\\n(Total Variance: {ev1+ev2:.1f}%)")
plt.xlabel(f"PC1 ({ev1:.1f}%) - Thermal Baseline")
plt.ylabel(f"PC2 ({ev2:.1f}%) - Rate of Change & Range")
plt.legend()
plt.show()""")
])
with open(NOTEBOOKS_DIR / "05_pca_visualization.ipynb", "w") as f:
    json.dump(nb05, f, indent=2)


# -------------------------------------------------------------
# 06_comparative_analysis.ipynb
# -------------------------------------------------------------
nb06 = make_notebook([
    md_cell("""# 06. Comparative Analysis, Evaluation & Limitations
**Course:** INT396 – Unsupervised Machine Learning  
**Problem Statement ID:** SIH26073

### Key Comparative Questions:
1. What is the consensus agreement between Isolation Forest and Local Outlier Factor?
2. What are the common anomalies ($IF \\cap LOF$) vs method-specific anomalies ($IF \\setminus LOF$, $LOF \\setminus IF$)?
3. What is the Jaccard similarity between anomaly sets across parameter configurations?
4. Do flagged observations correspond to sensor defects or legitimate extreme climate events?"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.comparison import run_comparative_pipeline
from src.evaluation import run_evaluation_pipeline

df_combined, df_grid, baseline = run_comparative_pipeline()
events_df, biases = run_evaluation_pipeline()
print("Comparative Evaluation Completed.")"""),
    code_cell("""# Baseline Metrics Summary
for k, v in baseline.items():
    print(f" - {k}: {v}")"""),
    code_cell("""# Parameter Sensitivity Grid
df_grid"""),
    code_cell("""# Verified Historical Extreme Events Audit
events_df[["Event", "Days", "IF_Flagged_Days", "LOF_Flagged_Days", "Consensus_Flagged_Days", "Assessment"]]"""),
    code_cell("""# Geographical Station Bias Breakdown
biases["station_bias"][["STATION", "Total_Records", "IF_Rate_Pct", "LOF_Rate_Pct", "Consensus_Rate_Pct"]]""")
])
with open(NOTEBOOKS_DIR / "06_comparative_analysis.ipynb", "w") as f:
    json.dump(nb06, f, indent=2)

print("All 6 Jupyter notebooks generated successfully in notebooks/")

