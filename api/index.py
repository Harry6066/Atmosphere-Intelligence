import sys
import os
from pathlib import Path

# Ensure Matplotlib writes only to /tmp on serverless environments
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app import create_app

app = create_app(os.environ.get("FLASK_ENV", "production"))
