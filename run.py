"""Atmosphere Intelligence Platform Runner.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Starts the Flask backend server.
"""

import os
import sys
import argparse
from pathlib import Path

# Set Matplotlib cache directory
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app import create_app

app = create_app(os.environ.get("FLASK_ENV", "development"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Atmosphere Intelligence Platform")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port number (default: 5000)")
    parser.add_argument("--no-debug", action="store_true", help="Disable Flask debug mode")
    args = parser.parse_args()

    print("=" * 75)
    print("ATMOSPHERE INTELLIGENCE — Weather Observation Anomaly Detection Platform")
    print("Course: INT396 (Unsupervised ML) | Problem Statement ID: SIH26073")
    print(f"Local Server URL: http://{args.host}:{args.port}")
    print(f"API Health Check: http://{args.host}:{args.port}/api/health")
    print(f"API Overview:     http://{args.host}:{args.port}/api/overview")
    print(f"API Summary:      http://{args.host}:{args.port}/api/dataset/summary")
    print(f"API Comparison:   http://{args.host}:{args.port}/api/comparison")
    print("=" * 75)

    app.run(host=args.host, port=args.port, debug=not args.no_debug)
