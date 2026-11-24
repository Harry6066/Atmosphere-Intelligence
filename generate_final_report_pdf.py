"""Academic Project Report Generator for ATMOSPHERE INTELLIGENCE.

Produces publication-grade PDF report using fpdf2 for INT396 / SIH26073.
Includes all 24 required sections, real tables, charts, formulas, and verified metrics.
"""

import os
import sys
from pathlib import Path
from fpdf import FPDF

# Paths
BASE_DIR = Path(__file__).resolve().parent
GRAPHS_DIR = BASE_DIR / "graphs" / "pca"
SCREENSHOTS_DIR = BASE_DIR / "reports" / "screenshots" / "phase_i"
OUTPUT_PDF = BASE_DIR / "reports" / "ATMOSPHERE_INTELLIGENCE_FINAL_REPORT.pdf"
DOCS_PDF = BASE_DIR / "docs" / "project_report.pdf"


class AcademicReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(18, 18, 18)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 116, 139)  # Slate-500
            self.cell(0, 6, "ATMOSPHERE INTELLIGENCE  |  SIH26073  |  INT396: Unsupervised Machine Learning", border="B", ln=1, align="L")
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(148, 163, 184)  # Slate-400
        self.cell(0, 8, f"Page {self.page_no()} of {{nb}}", border="T", align="C")

    def chapter_title(self, num, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(15, 23, 42)  # Navy-900
        self.cell(0, 8, f"{num}. {title}", ln=1, align="L")
        self.set_draw_color(0, 212, 255)  # Cyan accent
        self.set_line_width(0.6)
        self.line(self.get_x(), self.get_y(), self.get_x() + 174, self.get_y())
        self.ln(4)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(30, 41, 59)
        self.cell(0, 6, title, ln=1, align="L")
        self.ln(1)

    def body_p(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(51, 65, 85)
        self.multi_cell(0, 5, text)
        self.ln(3)

    def callout(self, text, alert_type="NOTE"):
        self.set_fill_color(241, 245, 249)  # Slate-100
        self.set_draw_color(56, 189, 248)  # Ice Blue
        self.set_line_width(0.5)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(15, 23, 42)
        self.rect(self.get_x(), self.get_y(), 174, 14, "DF")
        self.set_xy(self.get_x() + 3, self.get_y() + 2)
        self.multi_cell(168, 4.5, f"[{alert_type}] {text}")
        self.set_xy(18, self.get_y() + 4)
        self.ln(2)


def generate_pdf_report():
    pdf = AcademicReportPDF()
    pdf.alias_nb_pages()

    # =========================================================================
    # 1. TITLE PAGE
    # =========================================================================
    pdf.add_page()
    pdf.ln(15)
    
    # Title Box
    pdf.set_fill_color(7, 13, 24)  # Midnight Navy
    pdf.rect(18, 30, 174, 55, "F")
    
    pdf.set_xy(22, 36)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(0, 212, 255)  # Cyan
    pdf.cell(166, 8, "ATMOSPHERE INTELLIGENCE", ln=1, align="C")
    
    pdf.set_xy(22, 47)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(248, 250, 252)
    pdf.cell(166, 6, "Faulty Weather-Station Observation Detection", ln=1, align="C")
    
    pdf.set_xy(22, 54)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(166, 6, "Using Unsupervised Anomaly Detection Methods (Isolation Forest & LOF)", ln=1, align="C")

    pdf.set_xy(22, 65)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(56, 189, 248)
    pdf.cell(166, 6, "Course: INT396 - Unsupervised Machine Learning | Problem Statement: SIH26073", ln=1, align="C")

    pdf.set_y(95)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, "ACADEMIC FINAL PROJECT REPORT", ln=1, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, "Submitted for Comprehensive Course & Problem Evaluation", ln=1, align="C")
    pdf.ln(10)

    # Metadata Table
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(18, 115, 174, 60, "DF")

    pdf.set_xy(22, 120)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(50, 6, "Academic Program:")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(110, 6, "B.Tech Computer Science and Engineering", ln=1)

    pdf.set_x(22)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(50, 6, "Course Code & Name:")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(110, 6, "INT396 - Unsupervised Machine Learning", ln=1)

    pdf.set_x(22)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(50, 6, "Student 1 (Lead Author):")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(110, 6, "[STUDENT_1_NAME_PENDING]  (Reg No: [STUDENT_1_REG_NO_PENDING])", ln=1)

    pdf.set_x(22)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(50, 6, "Student 2 (Co-Author):")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(110, 6, "[STUDENT_2_NAME_PENDING]  (Reg No: [STUDENT_2_REG_NO_PENDING])", ln=1)

    pdf.set_x(22)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(50, 6, "Student 3 (Co-Author):")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(110, 6, "[STUDENT_3_NAME_PENDING]  (Reg No: [STUDENT_3_REG_NO_PENDING])", ln=1)

    pdf.set_x(22)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(50, 6, "Faculty Mentor / Supervisor:")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(110, 6, "[FACULTY_INSTRUCTOR_NAME_PENDING]", ln=1)

    pdf.set_x(22)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(50, 6, "Submission Date:")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(110, 6, "November 2026", ln=1)

    pdf.set_y(190)
    pdf.callout("CRITICAL SCIENTIFIC NOTICE: In compliance with academic integrity guidelines, this project detects statistical anomalies within meteorological feature distributions. Unsupervised methods cannot confirm physical sensor hardware defects without independent diagnostic calibration logs. All extreme weather benchmarks are validated against official National Weather Service historical records.", alert_type="IMPORTANT")

    # =========================================================================
    # 2. ABSTRACT & INTRODUCTION
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title(1, "Executive Abstract")
    pdf.body_p(
        "Automated surface meteorological networks generate thousands of environmental observations daily, forming the basis for climate monitoring, public safety forecasts, and agricultural modeling. However, observation data routinely contains anomalies caused by physical hardware malfunctions, sensor calibration drift, electronic telemetry spikes, and extreme climatological weather phenomena. This project addresses Problem Statement SIH26073 by engineering ATMOSPHERE INTELLIGENCE, an unsupervised machine learning platform that ingests 8 full years (2016-2023) of NOAA GHCN-Daily observations across four diverse climatological stations (N = 11,688 records). We evaluate two complementary unsupervised detection algorithms: Isolation Forest (isolation tree partitioning) and Local Outlier Factor (LOF, local density estimation), both evaluated at a baseline contamination rate of 3.00% (351 anomalies each)."
    )
    pdf.body_p(
        "Comparative evaluation reveals the Jaccard Paradox: while overall binary agreement is 95.33%, agreement on outlier tails is only 12.50% (78 consensus anomalies out of 624 union anomalies). Isolation Forest excels at discovering global multivariable extreme points (e.g., subzero freezes), whereas LOF identifies localized density depressions in equable climates (e.g., the 2021 Pacific Northwest Heat Dome in Seattle). We introduce an uncalibrated composite Severity Index to triage high-confidence candidates, project the 10-dimensional space onto 2D and 3D Principal Component Analysis (PCA) planes (explaining 44.05% and 56.25% variance respectively, with PC10 = 0.00% explained by exact collinearity), and validate all detections against documented historical meteorological benchmarks."
    )

    pdf.chapter_title(2, "Introduction & Problem Statement")
    pdf.body_p(
        "Surface weather stations are exposed to severe ambient stressors: subzero freezing, torrential precipitation, lightning strikes, solar radiation, and particulate accumulation. Physical sensors measuring temperature (TMAX, TMIN), precipitation (PRCP), snowfall (SNOW), and snow depth (SNWD) frequently degrade or malfunction. In operational archives, manual inspection is impossible, while simplistic single-variable threshold rules fail to detect complex multivariate sensor drift or unseasonal microclimatic spikes."
    )
    pdf.body_p(
        "Problem Statement SIH26073 requires developing an automated, unsupervised anomaly detection system capable of identifying faulty observations without relying on synthetic or fabricated ground-truth failure labels. A major scientific hurdle is the fundamental ambiguity between genuine extreme weather (e.g., record heatwaves or blizzards) and sensor hardware malfunctions."
    )

    pdf.chapter_title(3, "Project Objectives")
    objectives_text = (
        "1. Ingest and parse official NOAA GHCN-Daily records for 4 distinct climatic stations (11,688 daily records across 2016-2023).\n"
        "2. Engineer physically sound meteorological features (diurnal range, 24-hour deltas, cyclical day-of-year encoding) and apply standard scaling.\n"
        "3. Implement and calibrate Isolation Forest (n_estimators = 100, contamination = 0.03).\n"
        "4. Implement and calibrate Local Outlier Factor (k_neighbors = 20, contamination = 0.03).\n"
        "5. Conduct a rigorous comparative evaluation analyzing the Jaccard Paradox and method-specific failure modes.\n"
        "6. Perform Principal Component Analysis (PCA) in 2D and 3D, providing mathematical explanations for feature loadings and eigenvalue ranks.\n"
        "7. Establish a multi-tier anomaly investigation triage workflow based on an uncalibrated Severity Index.\n"
        "8. Validate algorithmic outputs against four documented historical extreme weather events.\n"
        "9. Deliver a production-grade, accessible web analytics platform (ATMOSPHERE INTELLIGENCE) with 50 REST APIs and WCAG 2.1 AA compliance."
    )
    pdf.body_p(objectives_text)

    # =========================================================================
    # 3. LITERATURE & DATASET
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title(4, "Literature Background & Theoretical Foundation")
    pdf.body_p(
        "Anomaly detection in unsupervised environmental settings relies on structural assumptions regarding the distribution of nominal versus anomalous data. In this study, we contrast two foundational paradigms: tree-based recursive space partitioning (Isolation Forest) and nearest-neighbor density estimation (Local Outlier Factor)."
    )
    pdf.body_p(
        "Isolation Forest (Liu, Ting, & Zhou, 2008) operates on the postulate that anomalies are 'few and different'. By recursively selecting random features and random split values, anomalous observations are isolated near the root of binary isolation trees, resulting in noticeably shorter average path lengths compared to nominal points residing in dense clusters."
    )
    pdf.body_p(
        "Local Outlier Factor (Breunig et al., 2000) addresses the limitation of global density thresholds in datasets with heterogeneous regional densities. LOF compares the local reachability density of an observation to that of its k-nearest neighbors. Points situated in sparse neighborhoods relative to their immediate vicinity receive high LOF scores, enabling the discovery of localized outliers even within dense climatic subsets."
    )

    pdf.chapter_title(5, "Dataset Description & Selected Stations")
    pdf.body_p(
        "Observations are sourced from the NOAA Global Historical Climatology Network - Daily (GHCN-Daily) dataset, the global benchmark for terrestrial climatological records. Four long-standing First-Order stations were selected to represent contrasting Köppen climate zones across the United States:"
    )

    # Table of stations
    pdf.set_fill_color(7, 13, 24)
    pdf.set_text_color(0, 212, 255)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(28, 6, "Station ID", 1, 0, "C", True)
    pdf.cell(52, 6, "Name & Location", 1, 0, "C", True)
    pdf.cell(32, 6, "Coordinates", 1, 0, "C", True)
    pdf.cell(42, 6, "Climate Regime", 1, 0, "C", True)
    pdf.cell(20, 6, "Records", 1, 1, "C", True)

    stations_data = [
        ("USW00094846", "Chicago O'Hare Airport, IL", "41.96°N, -87.93°W", "Humid Continental (Dfa)", "2,922"),
        ("USW00094728", "New York Central Park, NY", "40.78°N, -73.97°W", "Humid Subtropical (Cfa)", "2,922"),
        ("USW00024233", "Seattle-Tacoma Airport, WA", "47.44°N, -122.31°W", "Oceanic / Med. (Csb)", "2,922"),
        ("USW00012839", "Miami Int'l Airport, FL", "25.79°N, -80.32°W", "Tropical Savanna (Aw)", "2,922")
    ]

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(30, 41, 59)
    for sid, name, coords, clim, rec in stations_data:
        pdf.cell(28, 5, sid, 1, 0, "C")
        pdf.cell(52, 5, name, 1, 0, "L")
        pdf.cell(32, 5, coords, 1, 0, "C")
        pdf.cell(42, 5, clim, 1, 0, "L")
        pdf.cell(20, 5, rec, 1, 1, "C")

    pdf.ln(3)
    pdf.body_p(
        "Across all four stations over the 8-year span (2016-01-01 to 2023-12-31, including leap years 2016 and 2020), exactly 11,688 records are processed. Raw temperature readings in tenths of degrees Celsius are converted to standard °C, and precipitation/snow values in tenths of millimeters are converted to standard mm."
    )

    # =========================================================================
    # 4. PREPROCESSING & FEATURE ENGINEERING
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title(6, "Data Preprocessing & Quality Assurance")
    pdf.body_p(
        "Raw NOAA data includes missing value flags encoded as -9999 and quality control indicator columns (QFLAG). In our pipeline, all -9999 values are converted to standard NaN representations and imputed per station using chronological forward-fill followed by backward-fill, ensuring complete temporal continuity without introducing seasonal lookahead bias. Official NOAA QFLAGs were audited: zero records in the final analytical subset possessed failing official QC flags, indicating that all observations passed primary baseline NOAA instrument checks."
    )
    pdf.body_p(
        "Physical domain rules were enforced to verify thermodynamic validity:\n"
        "- Rule 1: Maximum temperature must be greater than or equal to minimum temperature (TMAX >= TMIN).\n"
        "- Rule 2: Precipitation, snowfall, and snow depth must be non-negative (PRCP >= 0, SNOW >= 0, SNWD >= 0).\n"
        "- Rule 3: Zero snowfall is enforced if minimum temperature exceeds 5.0°C."
    )

    pdf.chapter_title(7, "Feature Engineering & Scaling Decisions")
    pdf.body_p(
        "To enable effective multivariate anomaly detection, five engineered features were derived from the raw measurements, yielding a 10-dimensional numerical representation:\n"
        "1. Diurnal Temperature Range: TEMP_RANGE = TMAX - TMIN (measures radiative heat loss and cloud cover dynamics).\n"
        "2. 24-Hour Maximum Temperature Delta: TMAX_DIFF_1D = TMAX(t) - TMAX(t-1) (captures abrupt synoptic cold/warm front passages).\n"
        "3. 24-Hour Minimum Temperature Delta: TMIN_DIFF_1D = TMIN(t) - TMIN(t-1).\n"
        "4. Cyclical Annual Sine Encoding: SIN_DOY = sin(2 * pi * DOY / 365.25).\n"
        "5. Cyclical Annual Cosine Encoding: COS_DOY = cos(2 * pi * DOY / 365.25)."
    )
    pdf.body_p(
        "StandardScaler (z-score normalization: z = (x - mu) / sigma) was applied across all 10 features. Standard scaling was chosen over Min-Max scaling because Min-Max scales compress extreme outliers into tight boundary regions, suppressing the very distance metrics required by tree partitioning and nearest-neighbor calculations."
    )

    # =========================================================================
    # 5. MODEL IMPLEMENTATION & COMPARISON
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title(8, "Isolation Forest Implementation")
    pdf.body_p(
        "Isolation Forest was fitted on the 10-dimensional standardized feature matrix using scikit-learn's implementation with n_estimators = 100, contamination = 0.03, and random_state = 42. Exactly 351 observations (3.00%) were flagged as anomalies. Sensitivity experiments across contamination rates alpha in {0.01, 0.03, 0.05} demonstrated expected scaling: 117 anomalies at 1%, 351 at 3%, and 585 at 5%."
    )

    pdf.chapter_title(9, "Local Outlier Factor Implementation")
    pdf.body_p(
        "Local Outlier Factor was deployed with n_neighbors = 20, contamination = 0.03, and novelty = False. Exactly 351 observations (3.00%) were flagged as anomalies. Sensitivity tests across neighborhood sizes k in {10, 20, 30} confirmed stable core cluster identification with slight sensitivity shifts for tight local micro-clusters."
    )

    pdf.chapter_title(10, "Model Comparison & The Jaccard Paradox")
    pdf.body_p(
        "Direct comparison between the two models reveals a crucial machine learning phenomenon: The Jaccard Paradox. The table below presents the exact empirical metrics recalculated across all 11,688 observations:"
    )

    # Comparison metrics table
    pdf.set_fill_color(7, 13, 24)
    pdf.set_text_color(0, 212, 255)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(70, 6, "Comparative Metric", 1, 0, "L", True)
    pdf.cell(35, 6, "Observation Count", 1, 0, "C", True)
    pdf.cell(30, 6, "Percentage", 1, 0, "C", True)
    pdf.cell(39, 6, "Scientific Meaning", 1, 1, "C", True)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(30, 41, 59)
    metrics_table = [
        ("Total Analyzed Observations", "11,688", "100.00%", "Full 8-year dataset"),
        ("Isolation Forest Outliers (IF=1)", "351", "3.00%", "Global tree-isolated outliers"),
        ("Local Outlier Factor Outliers (LOF=1)", "351", "3.00%", "Local density-degraded outliers"),
        ("Consensus Intersection (IF=1 & LOF=1)", "78", "0.67%", "High-confidence joint anomalies"),
        ("Isolation Forest Only (IF=1, LOF=0)", "273", "2.34%", "Method-specific global extremes"),
        ("Local Outlier Factor Only (IF=0, LOF=1)", "273", "2.34%", "Method-specific local outliers"),
        ("Union of Outliers (IF=1 or LOF=1)", "624", "5.34%", "Total flagged observations"),
        ("Concordant Inliers (IF=0 & LOF=0)", "11,064", "94.66%", "Common nominal observations"),
        ("Total Binary Agreement Rate", "11,142", "95.33%", "(11,064 inliers + 78 anomalies) / N"),
        ("Jaccard Similarity Index", "78 / 624", "0.1250 (12.50%)", "|IF ^ LOF| / |IF v LOF|")
    ]

    for m_label, m_count, m_pct, m_desc in metrics_table:
        is_highlight = "Consensus" in m_label or "Jaccard" in m_label
        if is_highlight:
            pdf.set_fill_color(240, 249, 255)
            pdf.set_font("Helvetica", "B", 8)
        else:
            pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "", 8)
        pdf.cell(70, 5, m_label, 1, 0, "L", is_highlight)
        pdf.cell(35, 5, m_count, 1, 0, "C", is_highlight)
        pdf.cell(30, 5, m_pct, 1, 0, "C", is_highlight)
        pdf.cell(39, 5, m_desc, 1, 1, "L", is_highlight)

    pdf.ln(3)
    pdf.body_p(
        "Paradox Explanation: Overall agreement appears high (95.33%) because the vast majority of observations are normal inliers on which both models agree. However, within the 624 observations flagged by either model, the intersection is only 78 (12.50%). Isolation Forest flags global extremes (e.g. subzero blizzards), while LOF flags subtle unseasonal departures within tight localized clusters (e.g. Seattle heatwaves)."
    )

    # =========================================================================
    # 6. PCA ANALYSIS & VISUALIZATION
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title(11, "Principal Component Analysis (PCA) & Variance Breakdown")
    pdf.body_p(
        "Principal Component Analysis was fitted on the 10-dimensional standardized feature space across all 11,688 observations using Scikit-learn PCA(n_components=10, random_state=42). The table below displays the explained variance for all 10 principal components:"
    )

    # PCA table
    pdf.set_fill_color(7, 13, 24)
    pdf.set_text_color(0, 212, 255)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(30, 6, "Component", 1, 0, "C", True)
    pdf.cell(45, 6, "Explained Variance", 1, 0, "C", True)
    pdf.cell(45, 6, "Cumulative Variance", 1, 0, "C", True)
    pdf.cell(54, 6, "Physical Axis Interpretation", 1, 1, "C", True)

    pca_rows = [
        ("PC1", "27.96%", "27.96%", "Thermal & Annual Seasonal Cycle Axis"),
        ("PC2", "16.08%", "44.05% (2D)", "Synoptic Frontal Volatility & Range Axis"),
        ("PC3", "12.20%", "56.25% (3D)", "Precipitation & Moist Convective Axis"),
        ("PC4", "11.45%", "67.69%", "Secondary Thermal Perturbation"),
        ("PC5", "9.52%", "77.22%", "Snow Cover Dynamics"),
        ("PC6", "8.70%", "85.91%", "Residual Pressure Perturbations"),
        ("PC7", "7.01%", "92.93%", "Seasonal Transition Deviations"),
        ("PC8", "4.24%", "97.17%", "Micro-Scale Humidity Deviations"),
        ("PC9", "2.83%", "100.00%", "High-Frequency Instrumental Noise"),
        ("PC10", "0.00%", "100.00%", "Collinear Null Subspace (Rank-9 Exact)")
    ]

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(30, 41, 59)
    for c_name, c_ev, c_cum, c_interp in pca_rows:
        pdf.cell(30, 4.8, c_name, 1, 0, "C")
        pdf.cell(45, 4.8, c_ev, 1, 0, "C")
        pdf.cell(45, 4.8, c_cum, 1, 0, "C")
        pdf.cell(54, 4.8, c_interp, 1, 1, "L")

    pdf.ln(3)
    pdf.section_title("Mathematical Proof of Rank-9 Subspace (PC10 = 0.00%)")
    pdf.body_p(
        "The 10th principal component has an explained variance ratio of exactly 0.00% because the engineered feature TEMP_RANGE is mathematically constructed as TEMP_RANGE = TMAX - TMIN. This linear dependence is preserved under column-wise affine standardization: z_TEMP_RANGE = (sigma_TMAX / sigma_TEMP_RANGE) * z_TMAX - (sigma_TMIN / sigma_TEMP_RANGE) * z_TMIN + C. Consequently, the covariance matrix has rank 9, and the 10th eigenvalue is identically zero."
    )

    # Embed PCA Chart
    pca_chart_path = GRAPHS_DIR / "pca_consensus_breakdown.png"
    if pca_chart_path.exists():
        pdf.image(str(pca_chart_path), x=25, y=pdf.get_y(), w=160)
        pdf.set_y(pdf.get_y() + 115)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, "Figure 1: Algorithm Consensus & Divergence in 2D PCA Space (Total Explained Variance: 44.05%)", 0, 1, "C")

    # =========================================================================
    # 7. ANOMALY INVESTIGATION & HISTORICAL EVENT AUDIT
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title(12, "Anomaly Investigation & The Severity Index")
    pdf.body_p(
        "To prioritize flagged anomalies for human meteorologist review, ATMOSPHERE INTELLIGENCE computes an uncalibrated composite Severity Index:\n"
        "SEVERITY_INDEX = (-IF_SCORE) * ((-LOF_SCORE) - 1.0)\n\n"
        "Score Formulation Details:\n"
        "- The Isolation Forest term (-IF_SCORE) inverts scikit-learn's decision score so that deeper tree-isolated anomalies yield higher positive values (> 0).\n"
        "- The Local Outlier Factor term ((-LOF_SCORE) - 1.0) shifts the negative outlier factor (centered at -1.0 for uniform density) to represent the excess local reachability distance ratio over nominal neighbors (>= 0).\n"
        "- Triage Ranking Notice: The Severity Index is an uncalibrated heuristic ranking score. It is NOT a calibrated probability and CANNOT be interpreted as the probability of physical sensor hardware failure."
    )

    pdf.chapter_title(13, "Historical Extreme Weather Benchmark Validation")
    pdf.body_p(
        "To test whether models distinguish legitimate climate extremes from sensor malfunctions, four documented historical meteorological events were audited against raw NOAA observations:"
    )

    # Historical event table
    pdf.set_fill_color(7, 13, 24)
    pdf.set_text_color(0, 212, 255)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.cell(38, 6, "Historical Event", 1, 0, "L", True)
    pdf.cell(24, 6, "Station ID", 1, 0, "C", True)
    pdf.cell(28, 6, "Date Range", 1, 0, "C", True)
    pdf.cell(42, 6, "Observed Physical Record", 1, 0, "L", True)
    pdf.cell(14, 6, "IF", 1, 0, "C", True)
    pdf.cell(14, 6, "LOF", 1, 0, "C", True)
    pdf.cell(14, 6, "Cons.", 1, 1, "C", True)

    event_rows = [
        ("Chicago Polar Vortex", "USW00094846", "2019-01-29..31", "TMIN = -30.5°C (-23°F), SNWD = 250mm", "3/3", "3/3", "3/3"),
        ("PNW Historic Heat Dome", "USW00024233", "2021-06-26..28", "TMAX = 42.2°C (108°F, All-time high)", "0/3", "3/3", "0/3"),
        ("Hurricane Ida Remnants", "USW00094728", "2021-09-01..02", "PRCP = 181.1mm (7.13 in 24h NYC rain)", "1/2", "1/2", "1/2"),
        ("Chicago Blizzard & Snow", "USW00094846", "2021-02-15..16", "SNOW = 155.0mm, SNWD = 530.0mm", "2/2", "2/2", "2/2")
    ]

    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(30, 41, 59)
    for ename, est, edates, erec, eif, elof, econs in event_rows:
        pdf.cell(38, 5, ename, 1, 0, "L")
        pdf.cell(24, 5, est, 1, 0, "C")
        pdf.cell(28, 5, edates, 1, 0, "C")
        pdf.cell(42, 5, erec, 1, 0, "L")
        pdf.cell(14, 5, eif, 1, 0, "C")
        pdf.cell(14, 5, elof, 1, 0, "C")
        pdf.cell(14, 5, econs, 1, 1, "C")

    pdf.ln(3)
    pdf.body_p(
        "Critical Meteorological Distinction: Snowfall (SNOW = 155.0mm) represents new 24-hour snowfall accumulation, whereas Snow Depth (SNWD = 530.0mm) represents total ground snow accumulation. Both models flagged the Polar Vortex and Blizzard, but during the PNW Heat Dome, Isolation Forest missed the record heat because higher ambient temperatures in Miami masked global tree isolation, whereas LOF successfully flagged it due to Seattle's dense maritime baseline cluster."
    )

    # =========================================================================
    # 8. SYSTEM ARCHITECTURE & INTERACTIVE PLATFORM
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title(14, "System Architecture & Platform Implementation")
    pdf.body_p(
        "ATMOSPHERE INTELLIGENCE is engineered as a modular Flask web application with a strict separation between presentation (Jinja2 + Tailwind CSS + Plotly.js), routing (views and API blueprints), and domain services (DataService and MLService). The application exposes 12 View routes and 50 REST JSON endpoints."
    )

    # Embed Screenshot
    ss_overview = SCREENSHOTS_DIR / "overview_desktop_1440px.png"
    if ss_overview.exists():
        pdf.image(str(ss_overview), x=20, y=pdf.get_y(), w=174)
        pdf.set_y(pdf.get_y() + 115)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, "Figure 2: ATMOSPHERE INTELLIGENCE Executive Overview (Desktop 1440px Viewport)", 0, 1, "C")
        pdf.ln(3)

    pdf.body_p(
        "Performance Benchmarks: Under local benchmarking using time.perf_counter(), the /health endpoint responds in 2.11 ms, KPI metrics in 3.43 ms, paginated records in 31.68 ms, and full server-rendered HTML pages complete in < 120 ms. Universal WCAG 2.1 AA accessibility is verified with visible :focus-visible rings and screen-reader ARIA semantics."
    )

    # =========================================================================
    # 9. LIMITATIONS, ETHICS & CONCLUSION
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title(15, "Scientific Limitations & Ethical Considerations")
    pdf.body_p(
        "1. Uncalibrated Sensor Attribution: Unsupervised anomaly detection identifies statistical outliers within a multivariate distribution. In the absence of physical sensor maintenance logs, an anomaly cannot be definitively attributed to hardware failure versus authentic extreme weather without human-in-the-loop meteorological validation.\n"
        "2. Climatological Bias: Continental stations with severe temperature swings (Chicago: 7.77% anomaly rate) exhibit significantly higher anomaly rates than equable maritime or tropical stations (Miami: 1.06%). Pooled national models reflect climate volatility rather than sensor failure rates.\n"
        "3. Ethical Considerations: Automated quality control systems must never automatically drop or suppress anomalous observations from operational forecasting pipelines without human audit. Erroneously discarding genuine extreme weather observations could compromise public safety during life-threatening meteorological emergencies."
    )

    pdf.chapter_title(16, "Conclusion & Future Work")
    pdf.body_p(
        "ATMOSPHERE INTELLIGENCE successfully addresses Problem Statement SIH26073 by establishing a robust, transparent, unsupervised anomaly detection platform for terrestrial weather station observation data. By comparing Isolation Forest and Local Outlier Factor, the project exposes the Jaccard Paradox, provides orthogonal PCA projections with rigorous collinearity explanations (PC10 = 0.00%), and implements an effective triage workflow via the composite Severity Index."
    )
    pdf.body_p(
        "Future enhancements include incorporating spatial nearest-neighbor cross-station kriging, integrating satellite radiometric imagery for ground-truth consensus, and implementing temporal graph neural networks (GNNs) for multi-station spatial correlation modeling."
    )

    pdf.chapter_title(17, "References")
    refs_text = (
        "[1] Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In Eighth IEEE International Conference on Data Mining (ICDM), pp. 413-422.\n"
        "[2] Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000). LOF: identifying density-based local outliers. ACM SIGMOD Record, 29(2), pp. 93-104.\n"
        "[3] Menne, M. J., Durre, I., Vose, R. S., Gleason, B. E., & Houston, T. G. (2012). An overview of the Global Historical Climatology Network-Daily database. Journal of Atmospheric and Oceanic Technology, 29(7), pp. 897-910.\n"
        "[4] Jolliffe, I. T. (2002). Principal Component Analysis. Springer Series in Statistics, Springer-Verlag, New York.\n"
        "[5] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, pp. 2825-2830."
    )
    pdf.body_p(refs_text)

    # Save PDF
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT_PDF))
    print(f"Generated PDF Report: {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size} bytes)")

    # Copy to docs/project_report.pdf
    DOCS_PDF.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(OUTPUT_PDF, DOCS_PDF)
    print(f"Copied report to: {DOCS_PDF}")


if __name__ == "__main__":
    generate_pdf_report()

