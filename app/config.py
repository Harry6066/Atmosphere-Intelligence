"""Configuration settings for Atmosphere Intelligence Flask platform.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Defines paths, environment parameters, and runtime options.
"""

import os
from pathlib import Path


class BaseConfig:
    """Base application configuration."""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    SECRET_KEY = os.environ.get("SECRET_KEY", "atmosphere-secret-key-int396-sih26073")
    
    # Absolute paths to data and experimental outputs
    DATA_DIR = PROJECT_ROOT / "data"
    PROCESSED_DATA_PATH = DATA_DIR / "processed" / "ghcnd_processed.csv"
    RAW_DATA_PATH = DATA_DIR / "raw" / "ghcnd_raw_combined.csv"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    ANOMALY_RESULTS_DIR = OUTPUTS_DIR / "anomaly_results"
    COMPARISON_TABLES_DIR = OUTPUTS_DIR / "comparison_tables"
    EVALUATION_RESULTS_DIR = OUTPUTS_DIR / "evaluation_results"
    MODELS_DIR = PROJECT_ROOT / "models"
    GRAPHS_DIR = PROJECT_ROOT / "graphs"
    DOCS_DIR = PROJECT_ROOT / "docs"
    
    # Application identity
    APP_NAME = "ATMOSPHERE INTELLIGENCE"
    APP_SUBTITLE = "Weather Observation Anomaly Detection Platform"
    PROBLEM_ID = "SIH26073"
    COURSE = "INT396 – Unsupervised Machine Learning"
    
    JSON_SORT_KEYS = False


class DevelopmentConfig(BaseConfig):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class TestingConfig(BaseConfig):
    """Testing configuration."""
    DEBUG = False
    TESTING = True


class ProductionConfig(BaseConfig):
    """Production configuration."""
    DEBUG = False
    TESTING = False


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}

