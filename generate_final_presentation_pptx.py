"""Final Academic Presentation Generator for ATMOSPHERE INTELLIGENCE.

Produces a 15-slide widescreen (16:9) presentation using python-pptx for INT396 / SIH26073.
Includes ATMOSPHERE INTELLIGENCE design styling (navy/cyan theme), embedded real screenshots,
data cards, verified metrics, speaker notes, and student placeholders.
"""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Paths
BASE_DIR = Path(__file__).resolve().parent
GRAPHS_DIR = BASE_DIR / "graphs" / "pca"
SCREENSHOTS_DIR = BASE_DIR / "reports" / "screenshots" / "phase_i"
OUTPUT_PPTX = BASE_DIR / "reports" / "ATMOSPHERE_INTELLIGENCE_FINAL_PRESENTATION.pptx"
DOCS_PPTX = BASE_DIR / "docs" / "presentation.pptx"

# Color Palette (Midnight Navy & Cyan Accent)
COLOR_BG = RGBColor(7, 13, 24)         # #070d18 Midnight Navy
COLOR_SURFACE = RGBColor(12, 21, 39)    # #0c1527 Card Surface
COLOR_BORDER = RGBColor(30, 41, 59)     # #1e293b Slate-800
COLOR_CYAN = RGBColor(0, 212, 255)      # #00d4ff Electric Cyan
COLOR_ICE_BLUE = RGBColor(56, 189, 248) # #38bdf8 Ice Blue
COLOR_TEXT = RGBColor(248, 250, 252)    # #f8fafc Crisp White
COLOR_MUTED = RGBColor(148, 163, 184)   # #94a3b8 Muted Slate
COLOR_ACCENT_RED = RGBColor(239, 68, 68)# #ef4444 Anomaly Red


def apply_slide_background(slide):
    """Fill slide background with midnight navy."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG
    bg.line.color.rgb = COLOR_BG
    return bg


def add_slide_header(slide, title_text, category_text="ATMOSPHERE INTELLIGENCE | SIH26073"):
    """Render consistent top banner with category pill and title."""
    # Category / breadcrumb
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.3))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(9.5)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_CYAN

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.7))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(20)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT

    # Subtle cyan accent line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.733), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = COLOR_BORDER
    line.line.color.rgb = COLOR_BORDER


def add_speaker_note(slide, text):
    """Attach speaker notes to slide."""
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = text


def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: TITLE & TEAM DETAILS
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide1)

    # Main Card
    card = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(1.0), Inches(10.333), Inches(5.5))
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_SURFACE
    card.line.color.rgb = COLOR_BORDER

    tb = slide1.shapes.add_textbox(Inches(1.8), Inches(1.3), Inches(9.7), Inches(3.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "ATMOSPHERE INTELLIGENCE"
    p0.font.size = Pt(28)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_CYAN
    p0.alignment = PP_ALIGN.CENTER

    p1 = tf.add_paragraph()
    p1.text = "Faulty Weather-Station Observation Detection Using Anomaly Detection Methods"
    p1.font.size = Pt(16)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_TEXT
    p1.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = "Unsupervised Machine Learning Course Project (INT396)  |  Problem Statement: SIH26073"
    p2.font.size = Pt(12)
    p2.font.color.rgb = COLOR_ICE_BLUE
    p2.alignment = PP_ALIGN.CENTER

    # Author details box
    auth_box = slide1.shapes.add_textbox(Inches(2.0), Inches(3.6), Inches(9.333), Inches(2.5))
    tf_auth = auth_box.text_frame
    tf_auth.word_wrap = True

    pa1 = tf_auth.paragraphs[0]
    pa1.text = "Team Members:"
    pa1.font.bold = True
    pa1.font.size = Pt(11)
    pa1.font.color.rgb = COLOR_CYAN

    pa2 = tf_auth.add_paragraph()
    pa2.text = "* Student 1 (Lead Author): [STUDENT_1_NAME_PENDING]  |  Reg No: [STUDENT_1_REG_NO_PENDING]"
    pa2.font.size = Pt(10)
    pa2.font.color.rgb = COLOR_TEXT

    pa3 = tf_auth.add_paragraph()
    pa3.text = "* Student 2 (Co-Author):   [STUDENT_2_NAME_PENDING]  |  Reg No: [STUDENT_2_REG_NO_PENDING]"
    pa3.font.size = Pt(10)
    pa3.font.color.rgb = COLOR_TEXT

    pa4 = tf_auth.add_paragraph()
    pa4.text = "* Student 3 (Co-Author):   [STUDENT_3_NAME_PENDING]  |  Reg No: [STUDENT_3_REG_NO_PENDING]"
    pa4.font.size = Pt(10)
    pa4.font.color.rgb = COLOR_TEXT

    pa5 = tf_auth.add_paragraph()
    pa5.text = "Faculty Supervisor: [FACULTY_INSTRUCTOR_NAME_PENDING]  |  Department of CSE  |  November 2026"
    pa5.font.size = Pt(10)
    pa5.font.color.rgb = COLOR_MUTED

    add_speaker_note(slide1, "Welcome evaluators. Today we present ATMOSPHERE INTELLIGENCE, an unsupervised machine learning platform for detecting faulty weather station observations under Problem Statement SIH26073 for course INT396.")

    # =========================================================================
    # SLIDE 2: PROBLEM STATEMENT & CORE CHALLENGE
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide2)
    add_slide_header(slide2, "Problem Statement & Climatological Challenges")

    # Left Column: The Problem
    card_l = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    card_l.fill.solid()
    card_l.fill.fore_color.rgb = COLOR_SURFACE
    card_l.line.color.rgb = COLOR_BORDER

    tb_l = slide2.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.8))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True

    p = tf_l.paragraphs[0]
    p.text = "SIH26073 Problem Context"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    items_l = [
        "Terrestrial weather stations record crucial climate variables: TMAX, TMIN, PRCP, SNOW, and SNWD.",
        "Sensors face hostile environments: subzero freezing, icing, lightning, telemetry degradation, and electrical shorts.",
        "Manual inspection across thousands of stations is intractable.",
        "Simple hardcoded thresholds fail because an anomaly in Seattle (+40°C) is normal in Miami, and winter blizzards in Chicago have no counterpart in Florida."
    ]
    for itm in items_l:
        pi = tf_l.add_paragraph()
        pi.text = "• " + itm
        pi.font.size = Pt(10)
        pi.font.color.rgb = COLOR_TEXT

    # Right Column: The Core Scientific Dilemma
    card_r = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    card_r.fill.solid()
    card_r.fill.fore_color.rgb = COLOR_SURFACE
    card_r.line.color.rgb = COLOR_BORDER

    tb_r = slide2.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True

    p = tf_r.paragraphs[0]
    p.text = "The Scientific Dilemma: Outlier vs. Natural Disaster"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_ICE_BLUE

    items_r = [
        "Statistical Outlier != Physical Hardware Fault.",
        "An extreme reading could be a broken thermometer OR an authentic once-in-a-century climate catastrophe (e.g. Polar Vortex or Hurricane Ida).",
        "Zero Ground-Truth Failure Labels: NOAA datasets do not contain hardware failure diagnostic tags.",
        "Required Solution: An unsupervised multivariate detection architecture combining global tree isolation, local reachability density, consensus synthesis, and scientific human-in-the-loop triage."
    ]
    for itm in items_r:
        pi = tf_r.add_paragraph()
        pi.text = "• " + itm
        pi.font.size = Pt(10)
        pi.font.color.rgb = COLOR_TEXT

    add_speaker_note(slide2, "Slide 2 outlines the operational problem. Weather sensors face hardware degradation and environmental shocks. The key scientific challenge is distinguishing between sensor hardware failures and authentic extreme weather events without ground-truth labels.")

    # =========================================================================
    # SLIDE 3: MOTIVATION & OBJECTIVES
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide3)
    add_slide_header(slide3, "Project Motivation & Technical Objectives")

    card_obj = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    card_obj.fill.solid()
    card_obj.fill.fore_color.rgb = COLOR_SURFACE
    card_obj.line.color.rgb = COLOR_BORDER

    tb_obj = slide3.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(4.8))
    tf_obj = tb_obj.text_frame
    tf_obj.word_wrap = True

    p = tf_obj.paragraphs[0]
    p.text = "End-to-End Scientific Objectives"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = COLOR_CYAN

    objs = [
        ("1. Data Ingestion & Quality Control:", "Parse 11,688 multi-sensor records across 4 stations (2016-2023) from official NOAA GHCN-Daily archives, auditing quality control flags and physical thermodynamic constraints."),
        ("2. Feature Engineering & Scaling:", "Derive diurnal temperature ranges, 24-hour day-to-day deltas, and cyclical day-of-year trigonometric encodings, normalized via StandardScaler."),
        ("3. Unsupervised Dual-Model Deployment:", "Implement Isolation Forest (global tree isolation) and Local Outlier Factor (local reachability density), both calibrated to contamination = 0.03 (351 anomalies each)."),
        ("4. Comparative Synthesis & The Jaccard Paradox:", "Analyze model agreement and divergence, quantifying the Jaccard Paradox (95.33% binary agreement vs. 12.50% Jaccard similarity on anomaly tails)."),
        ("5. Orthogonal PCA Projection:", "Project 10-dimensional feature space onto 2D (44.05% variance) and 3D (56.25% variance) orthogonal planes, mathematically proving the rank-9 collinear null space (PC10 = 0.00%)."),
        ("6. Severity Index & Benchmark Audit:", "Establish an uncalibrated composite Severity Index for triage ranking, validating detections against four documented historical extreme weather events."),
        ("7. Accessible Commercial-Grade Web Platform:", "Deliver ATMOSPHERE INTELLIGENCE with 12 View routes, 50 REST JSON APIs, sub-50ms API latencies, and full WCAG 2.1 AA accessibility.")
    ]
    for o_title, o_desc in objs:
        pi = tf_obj.add_paragraph()
        pi.text = f"{o_title} {o_desc}"
        pi.font.size = Pt(9.5)
        pi.font.color.rgb = COLOR_TEXT

    add_speaker_note(slide3, "Our objectives span the entire pipeline from NOAA data ingestion and feature engineering to unsupervised modeling, comparative analysis, PCA projections, and benchmark validation.")

    # =========================================================================
    # SLIDE 4: DATASET & SELECTED STATIONS
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide4)
    add_slide_header(slide4, "NOAA GHCN-Daily Dataset & Climatological Stations")

    stations = [
        ("USW00094846", "Chicago O'Hare Int'l Airport", "41.96°N, -87.93°W", "Dfa - Humid Continental", "Severe freeze, blizzards, lake-effect snow", "2,922 days"),
        ("USW00094728", "New York Central Park", "40.78°N, -73.97°W", "Cfa - Humid Subtropical / Maritime", "Coastal storms, hurricane flash flooding", "2,922 days"),
        ("USW00024233", "Seattle-Tacoma Int'l Airport", "47.44°N, -122.31°W", "Csb - Oceanic / Mediterranean", "Tight equable maritime cluster, low variance", "2,922 days"),
        ("USW00012839", "Miami Int'l Airport", "25.79°N, -80.32°W", "Aw - Tropical Savanna", "Persistent tropical heat, zero snowfall", "2,922 days")
    ]

    for idx, (sid, sname, scoords, sclim, sfeat, sdays) in enumerate(stations):
        x = Inches(0.8 + idx * 2.95)
        w = Inches(2.85)
        card_s = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.6), w, Inches(5.2))
        card_s.fill.solid()
        card_s.fill.fore_color.rgb = COLOR_SURFACE
        card_s.line.color.rgb = COLOR_BORDER

        tb_s = slide4.shapes.add_textbox(x + Inches(0.15), Inches(1.8), w - Inches(0.3), Inches(4.8))
        tf_s = tb_s.text_frame
        tf_s.word_wrap = True

        p = tf_s.paragraphs[0]
        p.text = sid
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_CYAN

        p_name = tf_s.add_paragraph()
        p_name.text = sname
        p_name.font.bold = True
        p_name.font.size = Pt(9.5)
        p_name.font.color.rgb = COLOR_TEXT

        p_c = tf_s.add_paragraph()
        p_c.text = f"Coords: {scoords}\nClimate: {sclim}\nFeatures: {sfeat}\nCoverage: {sdays} (2016-2023)"
        p_c.font.size = Pt(8.5)
        p_c.font.color.rgb = COLOR_MUTED

    add_speaker_note(slide4, "We selected four primary First-Order NOAA stations covering 11,688 observations over 8 years, representing four distinct climate regimes from tropical Miami to continental Chicago and oceanic Seattle.")

    # =========================================================================
    # SLIDE 5: SYSTEM ARCHITECTURE
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide5)
    add_slide_header(slide5, "System Architecture & Engineering Workflow")

    card_arch = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    card_arch.fill.solid()
    card_arch.fill.fore_color.rgb = COLOR_SURFACE
    card_arch.line.color.rgb = COLOR_BORDER

    tb_arch = slide5.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(4.8))
    tf_arch = tb_arch.text_frame
    tf_arch.word_wrap = True

    p = tf_arch.paragraphs[0]
    p.text = "Modular Service-Oriented Web Platform"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    arch_layers = [
        ("Layer 1: Client Presentation", "Tailwind CSS Design Tokens, Vanilla ES6+ JS, Plotly.js Interactive Scientific Visualizations, Leaflet.js Interactive Station Map."),
        ("Layer 2: Web Routing & Views", "Flask Application Factory with 12 server-rendered Jinja2 View routes and 50 REST JSON API endpoints under /api/*."),
        ("Layer 3: Core Domain Services", "DataService (in-memory caching, indexing, multi-criteria filtering, streaming CSV export) and MLService (dynamic model execution, parameter validation, nearest-neighbor vector explanations)."),
        ("Layer 4: Modular Machine Learning Core", "src/ modules for data loading, thermodynamic physical rule checks, StandardScaler preprocessing, Isolation Forest, Local Outlier Factor, and 10D PCA projections."),
        ("Layer 5: Persistent Analytical Storage", "11,688-row processed CSV, precomputed model result CSVs, PCA loadings tables, and historical extreme event verification summaries.")
    ]
    for l_title, l_desc in arch_layers:
        pi = tf_arch.add_paragraph()
        pi.text = f"• {l_title}: {l_desc}"
        pi.font.size = Pt(9.5)
        pi.font.color.rgb = COLOR_TEXT

    add_speaker_note(slide5, "The platform is structured into five clean layers: client UI, Flask routing, domain services, modular machine learning algorithms, and persistent analytical storage.")

    # =========================================================================
    # SLIDE 6: DATA PREPROCESSING & QUALITY CONTROL
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide6)
    add_slide_header(slide6, "Data Preprocessing & Thermodynamic Validation")

    # Left Box: Preprocessing Steps
    card_p1 = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    card_p1.fill.solid()
    card_p1.fill.fore_color.rgb = COLOR_SURFACE
    card_p1.line.color.rgb = COLOR_BORDER

    tb_p1 = slide6.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.8))
    tf_p1 = tb_p1.text_frame
    tf_p1.word_wrap = True

    p = tf_p1.paragraphs[0]
    p.text = "Data Hygiene & Imputation"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    prep_steps = [
        "NOAA Unit Decoding: Raw temperatures in tenths of °C converted to standard °C; precipitation/snow in tenths of mm converted to mm.",
        "Sentinel Value Conversion: -9999 missing flags converted to NaN.",
        "Missing-Value Imputation: Imputed chronologically per-station using forward-fill followed by backward-fill (zero nulls in analytical feature space).",
        "Official QC Flags Audit: Audited PRCP_QFLAG, TMAX_QFLAG, etc. All analytical records passed baseline NOAA instrument QA checks."
    ]
    for s in prep_steps:
        pi = tf_p1.add_paragraph()
        pi.text = "• " + s
        pi.font.size = Pt(9.5)
        pi.font.color.rgb = COLOR_TEXT

    # Right Box: Physical Thermodynamic Rules
    card_p2 = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    card_p2.fill.solid()
    card_p2.fill.fore_color.rgb = COLOR_SURFACE
    card_p2.line.color.rgb = COLOR_BORDER

    tb_p2 = slide6.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_p2 = tb_p2.text_frame
    tf_p2.word_wrap = True

    p = tf_p2.paragraphs[0]
    p.text = "Physical Consistency Rule Enforcement"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_ICE_BLUE

    rules = [
        "Rule 1 (Thermodynamic Consistency): TMAX >= TMIN. Flag inverted temperature days where daily maximum is recorded below minimum.",
        "Rule 2 (Non-Negativity): PRCP >= 0, SNOW >= 0, SNWD >= 0. Enforce non-negative liquid and solid moisture accumulation.",
        "Rule 3 (Phase Transition Consistency): If TMIN > 5.0°C, SNOW must equal 0 (snowfall cannot accumulate at high surface temperatures).",
        "Zero Physical Violations in Final Set: Validates sensor baseline health prior to machine learning model ingestion."
    ]
    for r in rules:
        pi = tf_p2.add_paragraph()
        pi.text = "• " + r
        pi.font.size = Pt(9.5)
        pi.font.color.rgb = COLOR_TEXT

    add_speaker_note(slide6, "Slide 6 covers data hygiene and physical thermodynamic validation. We converted NOAA units, imputed missing values with forward/backward fill per station, and verified that zero physical rule violations exist.")

    # =========================================================================
    # SLIDE 7: FEATURE ENGINEERING & SCALING
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide7)
    add_slide_header(slide7, "10-Dimensional Meteorological Feature Engineering")

    card_feat = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    card_feat.fill.solid()
    card_feat.fill.fore_color.rgb = COLOR_SURFACE
    card_feat.line.color.rgb = COLOR_BORDER

    tb_feat = slide7.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(4.8))
    tf_feat = tb_feat.text_frame
    tf_feat.word_wrap = True

    p = tf_feat.paragraphs[0]
    p.text = "Feature Matrix Specification: x in R^10"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    feat_desc = [
        ("Raw Meteorological Features (5):", "TMAX (°C), TMIN (°C), PRCP (mm), SNOW (daily snowfall in mm), SNWD (ground snow accumulation in mm)."),
        ("Diurnal Temperature Range (1):", "TEMP_RANGE = TMAX - TMIN. Reflects radiative solar heating, cloud cover dampening, and diurnal atmospheric stability."),
        ("24-Hour Synoptic Deltas (2):", "TMAX_DIFF_1D and TMIN_DIFF_1D. Measures day-to-day air mass volatility and rapid frontal passages."),
        ("Cyclical Seasonal Encodings (2):", "SIN_DOY = sin(2*pi*DOY/365.25) and COS_DOY = cos(2*pi*DOY/365.25). Captures smooth annual solar cycles without artificial boundary discontinuities between Dec 31 and Jan 1."),
        ("Feature Scaling Decision (StandardScaler):", "z = (x - mu) / sigma applied across all 10 features. Standard scaling was chosen over Min-Max scaling because Min-Max artificially bounds extreme outliers into narrow margins, whereas z-score standardization preserves true Euclidean and tree-isolation distances in multivariate outlier space.")
    ]
    for f_title, f_body in feat_desc:
        pi = tf_feat.add_paragraph()
        pi.text = f"• {f_title} {f_body}"
        pi.font.size = Pt(9.5)
        pi.font.color.rgb = COLOR_TEXT

    add_speaker_note(slide7, "We mapped each observation to a 10-dimensional feature space including 5 raw measurements, diurnal range, 24-hour deltas, and cyclical day-of-year encodings, normalized using StandardScaler.")

    # =========================================================================
    # SLIDE 8: ISOLATION FOREST IMPLEMENTATION
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide8)
    add_slide_header(slide8, "Isolation Forest: Tree-Based Recursive Isolation")

    # Embed IF screenshot or graph
    if_chart = GRAPHS_DIR / "pca_isolation_forest.png"
    if if_chart.exists():
        slide8.shapes.add_picture(str(if_chart), Inches(0.8), Inches(1.6), width=Inches(6.2))

    card_if = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.6), Inches(5.333), Inches(5.2))
    card_if.fill.solid()
    card_if.fill.fore_color.rgb = COLOR_SURFACE
    card_if.line.color.rgb = COLOR_BORDER

    tb_if = slide8.shapes.add_textbox(Inches(7.4), Inches(1.8), Inches(4.9), Inches(4.8))
    tf_if = tb_if.text_frame
    tf_if.word_wrap = True

    p = tf_if.paragraphs[0]
    p.text = "Isolation Forest Baseline & Mechanics"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    if_points = [
        "Core Premise: Anomalies are 'few and different', requiring fewer random cuts to isolate into external leaves.",
        "Baseline Hyperparameters: n_estimators = 100, contamination = 0.03 (3.00%), random_state = 42.",
        "Flagged Anomalies: Exactly 351 observations (3.00% of 11,688 records).",
        "Score Formulation: Decision function returns s < 0 for anomalies, s > 0 for inliers.",
        "Sensitivity Analysis: Evaluated alpha in {0.01 (117 anomalies), 0.03 (351 anomalies), 0.05 (585 anomalies)}.",
        "Key Strength: Rapidly isolates global multidimensional extremes residing far from the central data centroid (e.g. subzero freezes, torrential rain spikes)."
    ]
    for pt in if_points:
        pi = tf_if.add_paragraph()
        pi.text = "• " + pt
        pi.font.size = Pt(9.5)
        pi.font.color.rgb = COLOR_TEXT

    add_speaker_note(slide8, "Isolation Forest constructs an ensemble of 100 random isolation trees. Anomalies isolate with short path lengths. At 3% contamination, it flags 351 global outliers.")

    # =========================================================================
    # SLIDE 9: LOCAL OUTLIER FACTOR (LOF) IMPLEMENTATION
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide9)
    add_slide_header(slide9, "Local Outlier Factor: Density-Based Anomaly Detection")

    lof_chart = GRAPHS_DIR / "pca_local_outlier_factor.png"
    if lof_chart.exists():
        slide9.shapes.add_picture(str(lof_chart), Inches(0.8), Inches(1.6), width=Inches(6.2))

    card_lof = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.6), Inches(5.333), Inches(5.2))
    card_lof.fill.solid()
    card_lof.fill.fore_color.rgb = COLOR_SURFACE
    card_lof.line.color.rgb = COLOR_BORDER

    tb_lof = slide9.shapes.add_textbox(Inches(7.4), Inches(1.8), Inches(4.9), Inches(4.8))
    tf_lof = tb_lof.text_frame
    tf_lof.word_wrap = True

    p = tf_lof.paragraphs[0]
    p.text = "Local Outlier Factor Baseline & Mechanics"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_ICE_BLUE

    lof_points = [
        "Core Premise: Compares the local reachability density (LRD) of a point against its k-nearest neighbors.",
        "Baseline Hyperparameters: n_neighbors = 20, contamination = 0.03 (3.00%), novelty = False.",
        "Flagged Anomalies: Exactly 351 observations (3.00% of 11,688 records).",
        "Score Formulation: Negative outlier factor (~ -1.0 for nominal density, < -1.5 for severe local outliers).",
        "Sensitivity Analysis: Tested k in {10, 20, 30}. Core anomaly set remained robust with slight shifts for micro-clusters.",
        "Key Strength: Successfully identifies localized density drops inside equable climates (e.g. Seattle heatwaves) that global tree methods overlook."
    ]
    for pt in lof_points:
        pi = tf_lof.add_paragraph()
        pi.text = "• " + pt
        pi.font.size = Pt(9.5)
        pi.font.color.rgb = COLOR_TEXT

    add_speaker_note(slide9, "Local Outlier Factor measures local reachability density. At k=20 and 3% contamination, it flags 351 anomalies, capturing localized density drops within tight regional clusters.")

    # =========================================================================
    # SLIDE 10: MODEL COMPARISON & THE JACCARD PARADOX
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide10)
    add_slide_header(slide10, "Model Comparison: The Jaccard Paradox")

    # Metrics Table
    card_comp = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    card_comp.fill.solid()
    card_comp.fill.fore_color.rgb = COLOR_SURFACE
    card_comp.line.color.rgb = COLOR_BORDER

    tb_comp = slide10.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(4.8))
    tf_comp = tb_comp.text_frame
    tf_comp.word_wrap = True

    p = tf_comp.paragraphs[0]
    p.text = "Empirical Model Agreement & Outlier Divergence (N = 11,688)"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    comp_items = [
        "Isolation Forest Anomalies: 351 (3.00%) | Local Outlier Factor Anomalies: 351 (3.00%)",
        "Consensus Intersection (Both Flagged): Exactly 78 observations (0.67% of total dataset).",
        "IF-Only Anomalies: 273 observations (2.34%) | LOF-Only Anomalies: 273 observations (2.34%)",
        "Union of All Outliers: 624 observations (5.34%) | Concordant Inliers: 11,064 observations (94.66%)",
        "Total Binary Agreement Rate: 95.33% ((11,064 inliers + 78 anomalies) / 11,688).",
        "Jaccard Similarity Index on Anomalies: Exactly 0.1250 (12.50% = 78 / 624).",
        "Scientific Finding (The Jaccard Paradox): The high 95.33% binary agreement is almost entirely driven by common normal inliers. On the 3% outlier tails, the algorithms disagree on 7 out of every 8 detections! This confirms that tree-based isolation and local density estimation isolate completely different geometric failure modes."
    ]
    for itm in comp_items:
        pi = tf_comp.add_paragraph()
        pi.text = "• " + itm
        pi.font.size = Pt(9.5)
        pi.font.color.rgb = COLOR_TEXT

    add_speaker_note(slide10, "Slide 10 highlights our key machine learning finding: The Jaccard Paradox. Overall agreement is 95.33%, but anomaly tail intersection is only 12.50%. The models capture complementary failure modes.")

    # =========================================================================
    # SLIDE 11: PCA 2D & 3D VISUALIZATION
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide11)
    add_slide_header(slide11, "PCA Projections & Collinear Null Space (Rank-9)")

    # Left: PCA Consensus Graph
    pca_cb = GRAPHS_DIR / "pca_consensus_breakdown.png"
    if pca_cb.exists():
        slide11.shapes.add_picture(str(pca_cb), Inches(0.8), Inches(1.6), width=Inches(6.2))

    # Right: Variance Breakdown & Proof
    card_pca = slide11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.6), Inches(5.333), Inches(5.2))
    card_pca.fill.solid()
    card_pca.fill.fore_color.rgb = COLOR_SURFACE
    card_pca.line.color.rgb = COLOR_BORDER

    tb_pca = slide11.shapes.add_textbox(Inches(7.4), Inches(1.8), Inches(4.9), Inches(4.8))
    tf_pca = tb_pca.text_frame
    tf_pca.word_wrap = True

    p = tf_pca.paragraphs[0]
    p.text = "PCA Variance & Subspace Proof"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    pca_points = [
        "Input Features: 10 standardized numerical variables.",
        "PC1 (27.96%): Thermal & Seasonal Axis (TMAX: +0.57, TMIN: +0.53, COS_DOY: -0.44).",
        "PC2 (16.08%): Synoptic Volatility & Range Axis (TMAX_DIFF_1D: +0.66, TMIN_DIFF_1D: +0.44, TEMP_RANGE: +0.43).",
        "PC3 (12.20%): Precipitation & Cloud Cover Axis (TMIN_DIFF_1D: +0.53, PRCP: +0.47, TEMP_RANGE: -0.45).",
        "Cumulative Explained Variance: 2D = 44.05% | 3D = 56.25%.",
        "Mathematical Proof of PC10 = 0.00%: Because TEMP_RANGE = TMAX - TMIN, exact linear collinearity is preserved under standardization. The covariance matrix has rank exactly 9, so the 10th eigenvalue is mathematically zero."
    ]
    for pt in pca_points:
        pi = tf_pca.add_paragraph()
        pi.text = "• " + pt
        pi.font.size = Pt(9.5)
        pi.font.color.rgb = COLOR_TEXT

    add_speaker_note(slide11, "Slide 11 presents our PCA results. 2D explains 44.05% of variance, 3D explains 56.25%. PC10 has exactly 0.00% variance because TEMP_RANGE equals TMAX minus TMIN, creating a rank-9 subspace.")

    # =========================================================================
    # SLIDE 12: ANOMALY INVESTIGATION & SEVERITY INDEX
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide12)
    add_slide_header(slide12, "Anomaly Investigation: The Severity Index")

    card_sev = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    card_sev.fill.solid()
    card_sev.fill.fore_color.rgb = COLOR_SURFACE
    card_sev.line.color.rgb = COLOR_BORDER

    tb_sev = slide12.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(4.8))
    tf_sev = tb_sev.text_frame
    tf_sev.word_wrap = True

    p = tf_sev.paragraphs[0]
    p.text = "Triage Prioritization: SEVERITY_INDEX = (-IF_SCORE) * ((-LOF_SCORE) - 1.0)"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    sev_pts = [
        "Rationale: To triage 624 flagged outliers, human auditors need a compound ranking metric that prioritizes observations with simultaneous global isolation and local density degradation.",
        "Sign Orientation 1 (-IF_SCORE): Scikit-learn's decision function is positive for nominal points and negative for outliers. Inverting the sign yields positive values (> 0) where larger numbers reflect deeper tree isolation.",
        "Sign Orientation 2 ((-LOF_SCORE) - 1.0): Scikit-learn's negative outlier factor is ~ -1.0 for uniform density and < -1.0 for outliers. Inverting and subtracting 1.0 isolates the excess local reachability distance over neighbors (>= 0).",
        "Compound Ranking: Multiplying both terms elevates consensus anomalies residing in the extreme tail of both algorithms.",
        "Critical Disclaimer: The Severity Index is an uncalibrated ranking heuristic. It is NOT a probability and does NOT measure the probability of physical sensor hardware failure."
    ]
    for pt in sev_pts:
        pi = tf_sev.add_paragraph()
        pi.text = "• " + pt
        pi.font.size = Pt(9.5)
        pi.font.color.rgb = COLOR_TEXT

    add_speaker_note(slide12, "The Severity Index compounds Isolation Forest tree isolation with LOF excess density degradation to rank candidates. It is strictly an uncalibrated triage ranking, not a failure probability.")

    # =========================================================================
    # SLIDE 13: SCIENTIFIC BENCHMARK VALIDATION & BIAS
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide13)
    add_slide_header(slide13, "Historical Event Validation & Climatological Bias")

    # Table of Historical Events
    card_ev = slide13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    card_ev.fill.solid()
    card_ev.fill.fore_color.rgb = COLOR_SURFACE
    card_ev.line.color.rgb = COLOR_BORDER

    tb_ev = slide13.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(4.8))
    tf_ev = tb_ev.text_frame
    tf_ev.word_wrap = True

    p = tf_ev.paragraphs[0]
    p.text = "Validation Against Documented Extreme Weather Events"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    ev_items = [
        "1. Chicago Polar Vortex (2019-01-29 to 2019-01-31): TMIN reached -30.5°C (-23°F). Flagged 3/3 by IF, 3/3 by LOF, 3/3 by Consensus. Validates detection of extreme cold waves.",
        "2. PNW Historic Heat Dome (2021-06-26 to 2021-06-28): Seattle TMAX reached 42.2°C (108°F, all-time record). Flagged 0/3 by IF, 3/3 by LOF, 0/3 by Consensus. Proves LOF's strength in detecting localized climate spikes.",
        "3. Hurricane Ida Remnants (2021-09-01 to 2021-09-02): NYC Central Park recorded 181.1mm (7.13 inches) rain in 24 hours. Flagged 1/2 by IF, 1/2 by LOF, 1/2 by Consensus. Validates torrential flash flood detection.",
        "4. Chicago Ground Blizzard (2021-02-15 to 2021-02-16): SNOW = 155.0mm (daily snowfall), SNWD = 530.0mm (snow depth). Flagged 2/2 by IF, 2/2 by LOF, 2/2 by Consensus. Accurately separates snowfall from snow depth.",
        "Climatological & Seasonal Bias: Continental Chicago experienced a 7.77% anomaly rate compared to only 1.06% in tropical Miami. Detections peaked in winter (9.66%) vs. autumn (3.78%), demonstrating that pooled models reflect climate volatility rather than sensor failure rates."
    ]
    for itm in ev_items:
        pi = tf_ev.add_paragraph()
        pi.text = "• " + itm
        pi.font.size = Pt(9.5)
        pi.font.color.rgb = COLOR_TEXT

    add_speaker_note(slide13, "We cross-referenced our models against four documented historical weather events. Both models caught the Polar Vortex and Blizzard. The PNW Heat Dome was caught only by LOF, illustrating its unique value.")

    # =========================================================================
    # SLIDE 14: CONCLUSION & FUTURE SCOPE
    # =========================================================================
    slide14 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide14)
    add_slide_header(slide14, "Conclusion & Future Scope")

    card_c1 = slide14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2))
    card_c1.fill.solid()
    card_c1.fill.fore_color.rgb = COLOR_SURFACE
    card_c1.line.color.rgb = COLOR_BORDER

    tb_c1 = slide14.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.8))
    tf_c1 = tb_c1.text_frame
    tf_c1.word_wrap = True

    p = tf_c1.paragraphs[0]
    p.text = "Project Conclusions"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    concl = [
        "Unsupervised Anomaly Detection Works: Successfully audited 11,688 multi-sensor observations across four distinct climate zones without artificial failure labels.",
        "Complementary Paradigms: Isolation Forest flags global extremes; Local Outlier Factor flags localized micro-cluster departures.",
        "Consensus Triage: Combining both models through consensus (78 high-confidence candidates) reduces false positives while preserving vital alerts.",
        "Production-Grade Web Platform: Delivered ATMOSPHERE INTELLIGENCE with 89 passing unit tests, 50 REST APIs, and responsive design down to 320px."
    ]
    for c in concl:
        pi = tf_c1.add_paragraph()
        pi.text = "• " + c
        pi.font.size = Pt(9.5)
        pi.font.color.rgb = COLOR_TEXT

    card_c2 = slide14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.2))
    card_c2.fill.solid()
    card_c2.fill.fore_color.rgb = COLOR_SURFACE
    card_c2.line.color.rgb = COLOR_BORDER

    tb_c2 = slide14.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.8))
    tf_c2 = tb_c2.text_frame
    tf_c2.word_wrap = True

    p = tf_c2.paragraphs[0]
    p.text = "Future Research Directions"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_ICE_BLUE

    fut = [
        "Spatial Kriging & Cross-Station Correlation: Incorporate physical distance and elevation to cross-verify adjacent weather stations in real-time.",
        "Satellite Radiometric Fusion: Combine ground observations with GOES/MODIS satellite cloud cover and surface temperature to confirm ground events.",
        "Spatio-Temporal Graph Neural Networks (GNNs): Model meteorological sensor networks as dynamic spatio-temporal graphs for end-to-end anomaly detection.",
        "Automated Sensor Diagnostics: Integrate internal sensor hardware telemetry (battery voltage, signal-to-noise ratio) for calibrated failure classification."
    ]
    for f in fut:
        pi = tf_c2.add_paragraph()
        pi.text = "• " + f
        pi.font.size = Pt(9.5)
        pi.font.color.rgb = COLOR_TEXT

    add_speaker_note(slide14, "In conclusion, ATMOSPHERE INTELLIGENCE successfully tackles problem SIH26073. Future work will integrate spatial cross-station kriging, satellite radiometric data, and graph neural networks.")

    # =========================================================================
    # SLIDE 15: REFERENCES & DEMONSTRATION LINKS
    # =========================================================================
    slide15 = prs.slides.add_slide(blank_layout)
    apply_slide_background(slide15)
    add_slide_header(slide15, "References & Demonstration Guide")

    card_ref = slide15.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2))
    card_ref.fill.solid()
    card_ref.fill.fore_color.rgb = COLOR_SURFACE
    card_ref.line.color.rgb = COLOR_BORDER

    tb_ref = slide15.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(4.8))
    tf_ref = tb_ref.text_frame
    tf_ref.word_wrap = True

    p = tf_ref.paragraphs[0]
    p.text = "Academic References & Demonstration Resources"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_CYAN

    refs = [
        ("Liu, Ting, & Zhou (2008):", "Isolation Forest. In Eighth IEEE International Conference on Data Mining (ICDM), pp. 413-422."),
        ("Breunig, Kriegel, Ng, & Sander (2000):", "LOF: Identifying Density-Based Local Outliers. In ACM SIGMOD Record, 29(2), pp. 93-104."),
        ("Menne et al. (2012):", "An Overview of the Global Historical Climatology Network-Daily Database. Journal of Atmospheric and Oceanic Technology, 29(7), pp. 897-910."),
        ("Jolliffe, I. T. (2002):", "Principal Component Analysis. Springer Series in Statistics, Springer-Verlag, New York."),
        ("Local Platform Access:", "http://127.0.0.1:5000  (Flask server running locally via python run.py)"),
        ("Source Code & Tests:", "89 passing automated tests running via pytest tests/ -v"),
        ("Demonstration Sequence:", "Overview -> Dataset Explorer -> EDA -> Data Quality -> Isolation Forest -> LOF -> Model Comparison -> PCA -> Anomaly Investigation -> Scientific Insights.")
    ]
    for r_title, r_desc in refs:
        pi = tf_ref.add_paragraph()
        pi.text = f"• {r_title} {r_desc}"
        pi.font.size = Pt(9.5)
        pi.font.color.rgb = COLOR_TEXT

    add_speaker_note(slide15, "Thank you for your time. The application is running locally at port 5000 with 89 passing tests. We now welcome questions and are ready for the live demonstration.")

    # Save presentation
    OUTPUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUTPUT_PPTX))
    print(f"Generated PowerPoint Presentation: {OUTPUT_PPTX} ({OUTPUT_PPTX.stat().st_size} bytes)")

    # Copy to docs/presentation.pptx
    DOCS_PPTX.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(OUTPUT_PPTX, DOCS_PPTX)
    print(f"Copied presentation to: {DOCS_PPTX}")


if __name__ == "__main__":
    create_presentation()

