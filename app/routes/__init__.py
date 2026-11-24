"""Routes package for Atmosphere Intelligence."""

from app.routes.api import api_bp
from app.routes.views import views_bp

__all__ = ["api_bp", "views_bp"]

