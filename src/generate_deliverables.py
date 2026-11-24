"""Automated University Deliverable Generator (PDF Report and PPTX Slides).

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Generates professional academic PDF report and PowerPoint presentation (.pptx)
using fpdf2 and python-pptx with embedded project charts, metrics, and tables.
"""

import os
import sys
from pathlib import Path
import json
import pandas as pd
from fpdf import FPDF
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_DIR / "docs"
GRAPHS_DIR = PROJECT_DIR / "graphs"
OUTPUTS_DIR = PROJECT_DIR / "outputs"


# ==============================================================================
# 1. PPTX PRESENTATION GENERATOR (15 SLIDES)
# ==============================================================================
def generate_presentation_pptx():
    """Build university-grade 15-slide PowerPoint deck."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    NAVY = RGBColor(24, 43, 73)
    CRIMSON = RGBColor(192, 57, 43)
    SLATE = RGBColor(52, 73, 94)
    LIGHT_GRAY = RGBColor(245, 247, 250)
    DARK_TEXT = RGBColor(33, 33, 33)
    MUTED_TEXT = RGBColor(100, 100, 100)
    
    def add_header(slide, title_text, category="INT396: UNSUPERVISED MACHINE LEARNING | SIH26073"):
        # Header category
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = CRIMSON
        
        # Main title
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.7))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = NAVY

    # --- SLIDE 1: TITLE SLIDE ---
    s1 = prs.slides.add_slide(blank_layout)
    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.0))
    tf1 = tbox.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "Faulty Weather-Station Observation Detection"
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = NAVY
    
    p2 = tf1.add_paragraph()
    p2.text = "Unsupervised Anomaly Detection using Isolation Forest, LOF & PCA"
    p2.font.size = Pt(20)
    p2.font.color.rgb = CRIMSON
    p2.space_before = Pt(14)
    
    p3 = tf1.add_paragraph()
    p3.text = "Problem Statement ID: SIH26073 | Course Code: INT396"
    p3.font.size = Pt(14)
    p3.font.color.rgb = SLATE
    p3.space_before = Pt(24)
    
    p4 = tf1.add_paragraph()
    p4.text = "Team Members: [Student Name 1] | [Student Name 2] | [Student Name 3]\nSupervisor / Instructor: [Course Instructor Name]"
    p4.font.size = Pt(13)
    p4.font.color.rgb = MUTED_TEXT
    p4.space_before = Pt(16)

    # --- SLIDE 2: INTRODUCTION & MOTIVATION ---
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Introduction & Climatological Motivation")
    tb2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    bullets2 = [
        ("Surface Meteorological Observation Networks:", True),
        ("Automated surface weather stations provide essential daily data for numerical weather prediction, disaster response, and climate change tracking.", False),
        ("The Problem of Measurement Corruption:", True),
        ("Physical sensors deployed in harsh outdoor environments experience hardware freezing, power surges, telemetry packet loss, and sensor calibration drifts.", False),
        ("Infeasibility of Manual Quality Control:", True),
        ("Global archives ingest millions of daily records. Manual human auditing is unscalable, requiring autonomous, unsupervised statistical screening.", False),
        ("Key Scientific Objective:", True),
        ("Develop an unsupervised anomaly detection framework to flag anomalous weather observations, contrast multiple algorithmic paradigms, and validate results via PCA projection.", False)
    ]
    for text, bold in bullets2:
        p = tf2.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(15 if bold else 13)
        p.font.color.rgb = NAVY if bold else DARK_TEXT
        p.space_before = Pt(10 if bold else 2)

    # --- SLIDE 3: PROBLEM STATEMENT ---
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Problem Statement: SIH26073")
    tb3 = s3.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    bullets3 = [
        ("Problem Statement Scope:", True),
        ("Weather observations contain unusual values caused by sensor failures, recording errors, or genuine extreme weather. The goal is to detect statistically unusual observations using Isolation Forest and LOF.", False),
        ("The Critical Dilemma in Climate Quality Control:", True),
        ("A fundamental challenge in meteorological machine learning is distinguishing between statistically unusual observations and confirmed sensor malfunctions. A 100-year cold wave produces mathematical outliers that represent legitimate historical climate events.", False),
        ("Evaluation Requirements:", True),
        ("Because real-world climate streams lack ground-truth defect labels, the system must evaluate detection consistency, parameter sensitivity, Jaccard similarity, and PCA 2D projections without fabricating labels.", False)
    ]
    for text, bold in bullets3:
        p = tf3.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(15 if bold else 13)
        p.font.color.rgb = NAVY if bold else DARK_TEXT
        p.space_before = Pt(12 if bold else 3)

    # --- SLIDE 4: OBJECTIVES ---
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Project Objectives & Scope")
    tb4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    bullets4 = [
        "1. Ingest official NOAA GHCN-Daily records across diverse climate regimes with exact QC flags.",
        "2. Parse tenths-of-unit scales (0.1°C, 0.1mm) and handle sentinel missing values (-9999).",
        "3. Engineer meteorologically sound features: Diurnal Temperature Range, cyclical Day-of-Year, and 1-day temporal differences.",
        "4. Implement and evaluate Isolation Forest across contamination settings (0.01, 0.03, 0.05).",
        "5. Implement and evaluate Local Outlier Factor across neighborhood parameters (k = 10, 20, 30).",
        "6. Quantify consensus agreement, algorithm-specific divergences, and Jaccard similarity.",
        "7. Project feature space into 2D PCA plane and analyze component eigenvector loadings.",
        "8. Validate detections against verified real-world extreme climate events (Polar Vortex, Hurricane Ida, Heat Dome).",
        "9. Deliver an interactive Streamlit web application with dynamic filters and downloadable results."
    ]
    for b in bullets4:
        p = tf4.add_paragraph()
        p.text = b
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_TEXT
        p.space_before = Pt(6)

    # --- SLIDE 5: DATASET DESCRIPTION ---
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Primary Dataset: NOAA GHCN-Daily")
    tb5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf5 = tb5.text_frame
    tf5.word_wrap = True
    bullets5 = [
        ("Source Archive:", True),
        ("NOAA National Centers for Environmental Information (NCEI) Global Historical Climatology Network - Daily.", False),
        ("Multi-Station Climatic Selection (2016-2023, 8 Continuous Years):", True),
        ("- USW00094728: New York City Central Park, NY (Humid Continental / Coastal)\n- USW00094846: Chicago O'Hare International Airport, IL (Severe Winter Continental)\n- USW00024233: Seattle Tacoma International Airport, WA (Marine West Coast)\n- USW00012839: Miami International Airport, FL (Tropical Savanna / Hurricane Exposure)", False),
        ("Dataset Scale:", True),
        ("Total Records: 11,688 daily observations (exactly 2,922 days per station) across 5 core elements (TMAX, TMIN, PRCP, SNOW, SNWD).", False)
    ]
    for text, bold in bullets5:
        p = tf5.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(14 if bold else 12)
        p.font.color.rgb = NAVY if bold else DARK_TEXT
        p.space_before = Pt(8 if bold else 2)

    # --- SLIDE 6: SYSTEM ARCHITECTURE ---
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Modular System Architecture")
    tb6 = s6.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf6 = tb6.text_frame
    tf6.word_wrap = True
    arch_steps = [
        ("Phase 1 - Ingestion & QC Flag Audit (src/data_loader.py):", True),
        ("Retrieves raw station CSVs; decodes measurement, quality (QFLAG), and source flags; resolves -9999 missing sentinels.", False),
        ("Phase 2 - Preprocessing & Scaling (src/data_preprocessing.py):", True),
        ("Converts tenths-of-units; conducts physical consistency checks (TMIN <= TMAX); executes feature scaling via StandardScaler.", False),
        ("Phase 3 - Unsupervised Detection Engines:", True),
        ("Runs Isolation Forest (src/isolation_forest.py) and Local Outlier Factor (src/lof_detection.py) across parameter grids.", False),
        ("Phase 4 - Comparative & Evaluation Analytics:", True),
        ("Calculates Jaccard similarity (src/comparison.py), fits 2D PCA projection (src/pca_analysis.py), and audits extreme events (src/evaluation.py).", False),
        ("Phase 5 - Interactive Dashboard (app/app.py):", True),
        ("Web application providing parameter exploration, consensus breakdown, PCA visualization, and CSV export.", False)
    ]
    for text, bold in arch_steps:
        p = tf6.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(13 if bold else 11.5)
        p.font.color.rgb = NAVY if bold else DARK_TEXT
        p.space_before = Pt(6 if bold else 1)

    # --- SLIDE 7: PREPROCESSING & FEATURE ENGINEERING ---
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Feature Engineering & Preprocessing Workflow")
    tb7 = s7.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.2))
    tf7 = tb7.text_frame
    tf7.word_wrap = True
    p7 = [
        ("Meteorological Feature Engineering:", True),
        ("1. Diurnal Temperature Range (TEMP_RANGE = TMAX - TMIN):\n   Detects inverted thermometers and sensor freeze.\n2. Cyclical Day-of-Year Encodings:\n   sin(2pi*DOY/365.25), cos(2pi*DOY/365.25) capture annual seasonal loop.\n3. 1-Day Temporal Shifts (Delta TMAX, Delta TMIN):\n   Detects rapid 24h electrical glitches and front passages.", False),
        ("Feature Scaling:", True),
        ("StandardScaler applied to all 10 continuous features, maintaining true distribution tails without artificial clipping.", False)
    ]
    for text, bold in p7:
        p = tf7.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(13 if bold else 11.5)
        p.font.color.rgb = NAVY if bold else DARK_TEXT
        p.space_before = Pt(8 if bold else 2)
        
    img_eda = GRAPHS_DIR / "eda" / "eda_temp_range_scatter.png"
    if img_eda.exists():
        s7.shapes.add_picture(str(img_eda), Inches(7.0), Inches(1.6), width=Inches(5.5))

    # --- SLIDE 8: ISOLATION FOREST ---
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Isolation Forest Implementation & Parameter Grid")
    tb8 = s8.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.2))
    tf8 = tb8.text_frame
    tf8.word_wrap = True
    p8 = [
        ("Algorithm Mechanics:", True),
        ("Isolates anomalous instances by recursively constructing orthogonal random feature splits. Anomalies require fewer partitions (shorter path lengths).", False),
        ("Anomaly Score Equation:", True),
        ("s(x, n) = 2^(-E[h(x)] / c(n))\nWhere c(n) is the average path length of unsuccessful BST searches.", False),
        ("Experimental Parameter Results (N = 11,688):", True),
        ("- Contamination 0.01: 117 anomalies (1.00%)\n- Contamination 0.03: 351 anomalies (3.00%) [Baseline]\n- Contamination 0.05: 585 anomalies (5.01%)", False)
    ]
    for text, bold in p8:
        p = tf8.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(13 if bold else 11.5)
        p.font.color.rgb = NAVY if bold else DARK_TEXT
        p.space_before = Pt(8 if bold else 2)
        
    img_if = GRAPHS_DIR / "isolation_forest" / "if_score_distributions.png"
    if img_if.exists():
        s8.shapes.add_picture(str(img_if), Inches(7.0), Inches(1.6), width=Inches(5.5))

    # --- SLIDE 9: LOCAL OUTLIER FACTOR ---
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Local Outlier Factor (LOF) Density Detection")
    tb9 = s9.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.2))
    tf9 = tb9.text_frame
    tf9.word_wrap = True
    p9 = [
        ("Algorithm Mechanics:", True),
        ("Measures the local density deviation of an observation relative to its k-nearest neighbors using local reachability density (lrd).", False),
        ("Local Outlier Factor Equation:", True),
        ("LOF_k(p) = (1/k) * sum(lrd_k(o) / lrd_k(p))\nIdentifies observations residing in substantially sparser local regions.", False),
        ("Experimental Parameter Results (c = 0.03):", True),
        ("- k = 10 neighbors: 351 anomalies (Offset: -1.356)\n- k = 20 neighbors: 351 anomalies (Offset: -1.363) [Baseline]\n- k = 30 neighbors: 351 anomalies (Offset: -1.377)", False)
    ]
    for text, bold in p9:
        p = tf9.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(13 if bold else 11.5)
        p.font.color.rgb = NAVY if bold else DARK_TEXT
        p.space_before = Pt(8 if bold else 2)
        
    img_lof = GRAPHS_DIR / "lof" / "lof_score_distributions.png"
    if img_lof.exists():
        s9.shapes.add_picture(str(img_lof), Inches(7.0), Inches(1.6), width=Inches(5.5))

    # --- SLIDE 10: COMPARATIVE RESULTS ---
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Comparative Analysis: Isolation Forest vs. LOF")
    tb10 = s10.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.2))
    tf10 = tb10.text_frame
    tf10.word_wrap = True
    p10 = [
        ("Consensus Breakdown (Baseline IF c=0.03 vs LOF k=20):", True),
        ("- Total Observations: 11,688\n- Consensus Anomalies (IF & LOF): 78 observations\n- Isolation Forest Only: 273 (captures global extremes)\n- LOF Only: 273 (captures local density outliers)\n- Agreement Rate: 95.33%\n- Jaccard Similarity Index: 0.1250", False),
        ("Key Algorithmic Finding:", True),
        ("Isolation Forest prioritizes global distribution tails, while LOF detects subtle isolated combinations. They exhibit complementary strengths.", False)
    ]
    for text, bold in p10:
        p = tf10.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(13 if bold else 11.5)
        p.font.color.rgb = NAVY if bold else DARK_TEXT
        p.space_before = Pt(8 if bold else 2)
        
    img_comp = GRAPHS_DIR / "comparisons" / "consensus_category_breakdown.png"
    if img_comp.exists():
        s10.shapes.add_picture(str(img_comp), Inches(7.0), Inches(1.6), width=Inches(5.5))

    # --- SLIDE 11: PCA 2D VISUALIZATION ---
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "PCA 2D Dimensionality Reduction & Loadings")
    tb11 = s11.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.2))
    tf11 = tb11.text_frame
    tf11.word_wrap = True
    p11 = [
        ("Role of PCA in Project:", True),
        ("PCA is a linear dimensionality reduction and visualization technique, NOT the anomaly detection model. It projects 10D space onto 2D.", False),
        ("Explained Variance Ratios:", True),
        ("- PC1 (27.96%): Governed by thermal baseline (TMAX +0.572, TMIN +0.533, cos(DOY) -0.439).\n- PC2 (16.08%): Governed by volatility & diurnal range (Delta TMAX +0.662, TEMP_RANGE +0.432).\n- Cumulative 2D Variance: 44.05%", False)
    ]
    for text, bold in p11:
        p = tf11.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(13 if bold else 11.5)
        p.font.color.rgb = NAVY if bold else DARK_TEXT
        p.space_before = Pt(8 if bold else 2)
        
    img_pca = GRAPHS_DIR / "pca" / "pca_consensus_breakdown.png"
    if img_pca.exists():
        s11.shapes.add_picture(str(img_pca), Inches(7.0), Inches(1.6), width=Inches(5.5))

    # --- SLIDE 12: EVALUATION & FALSE POSITIVES ---
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Responsible Evaluation: Statistical Anomaly != Sensor Fault")
    tb12 = s12.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf12 = tb12.text_frame
    tf12.word_wrap = True
    p12 = [
        ("Audit Against Documented Historical Climate Events:", True),
        ("1. 2019 Chicago Polar Vortex Freeze (Jan 29-31, 2019; TMIN = -30.5°C): Flagged 3/3 days by BOTH models.\n2. 2021 NYC Hurricane Ida Flash Floods (Sep 1-2, 2021; PRCP > 180mm): Flagged by BOTH models.\n3. 2021 Chicago Blizzard & Snowpack (Feb 15-16, 2021; SNWD = 560mm): Flagged 2/2 days by BOTH models.\n4. 2021 Pacific Northwest Heat Dome (Seattle TMAX = 42.2°C): Flagged by LOF as a local density anomaly.", False),
        ("Critical Scientific Conclusion:", True),
        ("These observations are validated real-world climate records, NOT hardware failures. Unsupervised anomaly models isolate statistical rarities; human-in-the-loop meteorological triage is mandatory.", False)
    ]
    for text, bold in p12:
        p = tf12.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(14 if bold else 12)
        p.font.color.rgb = NAVY if bold else DARK_TEXT
        p.space_before = Pt(8 if bold else 2)

    # --- SLIDE 13: BIAS & LIMITATIONS ---
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Geographical & Seasonal Bias in Climate ML")
    tb13 = s13.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf13 = tb13.text_frame
    tf13.word_wrap = True
    p13 = [
        ("Geographical Station Bias:", True),
        ("High-variance continental stations (Chicago) yielded a 7.77% IF anomaly rate, whereas equable tropical stations (Miami) registered only 0.24%. Pooling disparate climate regimes into a single model creates severe regional bias.", False),
        ("Seasonal Variance Bias:", True),
        ("Winter conditions produced 9.66% IF anomaly flags compared to only 0.17% in Summer due to sub-zero freezing dispersion.", False),
        ("Recommended Academic Mitigation:", True),
        ("Train station-specific or regional-cluster models with localized seasonal baselines rather than global pooled thresholds.", False)
    ]
    for text, bold in p13:
        p = tf13.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(14 if bold else 12)
        p.font.color.rgb = NAVY if bold else DARK_TEXT
        p.space_before = Pt(8 if bold else 2)

    # --- SLIDE 14: INDIVIDUAL CONTRIBUTIONS ---
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Individual Member Contributions")
    tb14 = s14.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf14 = tb14.text_frame
    tf14.word_wrap = True
    members = [
        ("Member 1 ([Student 1 Name] - Reg No: [Reg No 1]):", True),
        ("- NOAA GHCN-Daily ingestion pipeline and schema validation (src/data_loader.py).\n- Unit decoding (tenths-of-units) and physical QC rule auditing (src/data_preprocessing.py).\n- Automated Pytest test suite implementation (tests/test_data.py, tests/test_preprocessing.py).", False),
        ("Member 2 ([Student 2 Name] - Reg No: [Reg No 2]):", True),
        ("- Feature engineering of diurnal range and cyclical Day-of-Year (src/feature_engineering.py).\n- Isolation Forest implementation and multi-parameter experiments (src/isolation_forest.py).\n- PCA 2D projection, explained variance, and component loading extraction (src/pca_analysis.py).", False),
        ("Member 3 ([Student 3 Name] - Reg No: [Reg No 3]):", True),
        ("- Local Outlier Factor implementation and neighborhood parameter experiments (src/lof_detection.py).\n- Comparative analysis engine and Jaccard similarity matrices (src/comparison.py).\n- Interactive Streamlit application (app/app.py) and historical event evaluation (src/evaluation.py).", False)
    ]
    for text, bold in members:
        p = tf14.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(14 if bold else 12)
        p.font.color.rgb = NAVY if bold else DARK_TEXT
        p.space_before = Pt(6 if bold else 1)

    # --- SLIDE 15: CONCLUSION & REFERENCES ---
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Conclusion & Academic References")
    tb15 = s15.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf15 = tb15.text_frame
    tf15.word_wrap = True
    p15 = [
        ("Summary of Deliverables:", True),
        ("1. Fully functional, modular unsupervised anomaly detection codebase with 13 automated unit tests.\n2. Ingested 11,688 NOAA records across 4 stations (2016-2023) with reproducible seeds.\n3. Verified complementary nature of Isolation Forest (global extremes) and LOF (local density).\n4. 44.05% variance explained in 2D PCA space with documented physical loadings.\n5. Interactive Streamlit dashboard ready for live academic demonstration.", False),
        ("Primary References:", True),
        ("1. Liu, Ting & Zhou (2008). Isolation Forest. IEEE ICDM.\n2. Breunig, Kriegel, Ng & Sander (2000). LOF: Identifying Density-Based Local Outliers. ACM SIGMOD.\n3. Menne et al. (2012). Overview of GHCN-Daily Database. J. Atmos. Oceanic Technol.", False)
    ]
    for text, bold in p15:
        p = tf15.add_paragraph()
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(14 if bold else 12)
        p.font.color.rgb = NAVY if bold else DARK_TEXT
        p.space_before = Pt(8 if bold else 2)

    out_pptx = DOCS_DIR / "presentation.pptx"
    prs.save(str(out_pptx))
    print(f"Presentation saved successfully to: {out_pptx}")


# ==============================================================================
# 2. PDF REPORT GENERATOR
# ==============================================================================
class AcademicReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 7, "INT396: Unsupervised Machine Learning | Problem ID: SIH26073", border=0, align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", border=0, align="C")


def generate_report_pdf():
    """Build university-grade academic PDF report."""
    pdf = AcademicReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title Section
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(24, 43, 73)
    pdf.multi_cell(0, 8, "Faulty Weather-Station Observation Detection Using Anomaly Detection Methods", align="C")
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(192, 57, 43)
    pdf.cell(0, 6, "Problem Statement ID: SIH26073 | Course Code: INT396", ln=True, align="C")
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 5, "Submitted for Academic Evaluation in INT396: Unsupervised Machine Learning\nGroup Members: [Student Name 1] | [Student Name 2] | [Student Name 3]\nFaculty Instructor: [Course Instructor Name]", align="C")
    pdf.ln(8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    
    def section_heading(title):
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(24, 43, 73)
        pdf.cell(0, 7, title, ln=True)
        pdf.ln(1)
        
    def sub_heading(title):
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.set_text_color(192, 57, 43)
        pdf.cell(0, 6, title, ln=True)
        pdf.ln(1)

    def body_text(text):
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, text)
        pdf.ln(3)

    # 1. Introduction & Problem Statement
    section_heading("1. Introduction and Problem Statement")
    body_text(
        "Automated ground weather stations deployed globally by meteorological agencies form the foundation of weather forecasting, flood warnings, and climate research. "
        "These stations record daily maximum temperature (TMAX), minimum temperature (TMIN), precipitation (PRCP), snowfall (SNOW), and snow depth (SNWD). "
        "However, sensor stations suffer from environmental degradation, freezing, electrical faults, and transmission drops. "
        "Under Problem Statement SIH26073, this project develops an unsupervised quality control pipeline that identifies statistically unusual weather observations using "
        "Isolation Forest and Local Outlier Factor (LOF), evaluated through Principal Component Analysis (PCA) 2D projection and historical climate case studies."
    )
    
    # 2. Objectives
    section_heading("2. Project Objectives")
    body_text(
        "- Ingest official NOAA GHCN-Daily records with quality flags and resolve missing value sentinels (-9999).\n"
        "- Conduct physical audits (TMIN <= TMAX) and engineer features: Diurnal Temperature Range (TEMP_RANGE) and cyclical Day-of-Year.\n"
        "- Train Isolation Forest across contamination settings (0.01, 0.03, 0.05).\n"
        "- Train Local Outlier Factor across neighborhood configurations (k = 10, 20, 30).\n"
        "- Perform comparative analysis: calculate Jaccard similarity, consensus overlap, and agreement percentages.\n"
        "- Project observations onto 2D PCA plane and analyze component eigenvector loadings.\n"
        "- Distinguish statistical anomalies from legitimate extreme weather events (e.g. Polar Vortex, Hurricane Ida).\n"
        "- Deploy an interactive Streamlit dashboard for project demonstration."
    )
    
    # 3. Dataset & Preprocessing
    section_heading("3. Dataset Description and Preprocessing")
    body_text(
        "The project utilizes official NOAA GHCN-Daily records across 4 stations selected for distinct climate zones: Central Park NY, Chicago O'Hare IL, Seattle Tacoma WA, and Miami FL. "
        "Covering 8 continuous years (2016-2023), the dataset comprises 11,688 daily observations (2,922 per station). "
        "Tenths-of-units were normalized to standard SI units (Celsius and mm). Missing precipitation and snow values in warm climates were imputed with 0.0 mm, "
        "and missing temperatures were forward-filled preserving temporal autocorrelation. All 10 engineered features were standardized using StandardScaler."
    )
    
    # 4. Experimental Results Table
    section_heading("4. Experimental Results and Algorithm Comparison")
    body_text(
        "Across the 11,688 observations, Isolation Forest and LOF were evaluated across multiple parameter configurations. "
        "At baseline settings (IF contamination = 0.03, LOF k = 20), both algorithms flagged exactly 351 anomalies (3.00%), achieving an overall agreement rate of 95.33% "
        "with 78 consensus anomalies (IF and LOF intersection) and a Jaccard similarity of 0.1250."
    )
    
    # Add parameter matrix text
    sub_heading("Cross-Parameter Sensitivity Matrix:")
    pdf.set_font("Courier", "", 8.5)
    matrix_text = (
        "IF_Contam  LOF_k  IF_Anom  LOF_Anom  Consensus  Agreement%  Jaccard\n"
        "  0.01      10      117      351        23        96.39%     0.0517\n"
        "  0.01      20      117      351        30        96.51%     0.0685\n"
        "  0.03      10      351      351        54        94.92%     0.0833\n"
        "  0.03      20      351      351        78        95.33%     0.1250\n"
        "  0.05      20      585      351       117        93.99%     0.1429\n"
        "  0.05      30      585      351       132        94.25%     0.1642"
    )
    pdf.multi_cell(0, 4.5, matrix_text)
    pdf.ln(3)

    # 5. PCA Projection
    section_heading("5. Principal Component Analysis (PCA) Visualization")
    body_text(
        "PCA reduced the 10-dimensional scaled feature space to 2 principal components, capturing 44.05% of total variance (PC1 = 27.96%, PC2 = 16.08%). "
        "PC1 loadings were dominated by baseline thermodynamic temperature (TMAX +0.572, TMIN +0.533, and winter cycle cos(DOY) -0.439). "
        "PC2 loadings were dominated by dynamic rate of change and diurnal swing (Delta TMAX +0.662, TEMP_RANGE +0.432). "
        "The 2D visualization reveals that consensus anomalies populate the outer boundary of the distribution."
    )
    
    # 6. Evaluation & Climate Events
    section_heading("6. Evaluation: Statistical Anomaly vs. Confirmed Fault")
    body_text(
        "Crucially, our evaluation audited flagged observations against officially documented historical extreme climate events:\n"
        "1. Chicago Polar Vortex (Jan 29-31, 2019; TMIN = -30.5 C): Flagged 3 out of 3 days by both models.\n"
        "2. Hurricane Ida Rainstorm (Sep 1-2, 2021; PRCP > 180 mm): Flagged by both models.\n"
        "3. Chicago Blizzard (Feb 15-16, 2021; SNWD = 560 mm): Flagged 2 out of 2 days by both models.\n"
        "4. Pacific Northwest Heat Dome (Seattle TMAX = 42.2 C): Flagged by LOF as a local density anomaly.\n\n"
        "Finding: These observations are confirmed legitimate extreme weather events, not hardware sensor malfunctions. "
        "Unsupervised anomaly detection identifies statistical outliers; human meteorological triage remains essential."
    )
    
    # 7. Limitations & Bias
    section_heading("7. Geographical Bias & Limitations")
    body_text(
        "Auditing detection rates across stations revealed significant geographical bias: Chicago O'Hare produced a 7.77% IF anomaly rate, "
        "whereas Miami produced only 0.24%. Winter observations produced a 9.66% IF anomaly rate compared to 0.17% in Summer. "
        "Pooling disparate climate regimes into a single global model causes milder regions to be under-flagged and variable regions to be over-flagged. "
        "Localized regional models are recommended for operational deployment."
    )
    
    # 8. Individual Contributions
    section_heading("8. Individual Contributions of Group Members")
    body_text(
        "- Member 1 ([Student 1 Name]): Data ingestion, unit conversion, physical QC rule audits, automated unit tests.\n"
        "- Member 2 ([Student 2 Name]): Feature engineering, Isolation Forest parameter experiments, PCA projection and loadings.\n"
        "- Member 3 ([Student 3 Name]): LOF density-based reachability, comparative analysis engine, Streamlit dashboard, event evaluation."
    )

    out_pdf = DOCS_DIR / "project_report.pdf"
    pdf.output(str(out_pdf))
    print(f"Academic report PDF saved successfully to: {out_pdf}")


if __name__ == "__main__":
    generate_presentation_pptx()
    generate_report_pdf()

