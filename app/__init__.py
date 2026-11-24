"""Atmosphere Intelligence Flask Application Factory.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Configures Flask instance, initializes DataService & MLService, and registers blueprints.
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask
from flask_cors import CORS

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.config import config_by_name, BaseConfig
from app.services.data_service import DataService
from app.services.ml_service import MLService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s"
)
logger = logging.getLogger("app")


def create_app(config_name: str = "default") -> Flask:
    """Flask application factory pattern.
    
    Parameters
    ----------
    config_name : str, default 'default'
        Name of configuration profile ('development', 'testing', 'production').
        
    Returns
    -------
    Flask
        Configured Flask application instance.
    """
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates"
    )
    
    # Load configuration
    config_cls = config_by_name.get(config_name, BaseConfig)
    app.config.from_object(config_cls)
    
    # Enable Cross-Origin Resource Sharing
    CORS(app)
    
    # Preload and validate DataService and MLService singletons
    with app.app_context():
        try:
            ds = DataService.get_instance(config_cls)
            ml = MLService.get_instance(ds)
            logger.info("Atmosphere Intelligence services initialized successfully.")
        except Exception as exc:
            logger.error("Failed to initialize DataService: %s", exc)
            raise
            
    # Register blueprints
    from app.routes.views import views_bp
    from app.routes.api import api_bp
    
    app.register_blueprint(views_bp)
    app.register_blueprint(api_bp)
    
    logger.info("Registered Flask Blueprints: 'views' and 'api' (/api)")
    return app
