import sys
import os
from pathlib import Path

# Ensure Matplotlib writes only to /tmp on serverless environments
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app import create_app

app = create_app(os.environ.get("FLASK_ENV", "production"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
