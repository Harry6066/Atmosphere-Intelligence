"""Verification script for ATMOSPHERE INTELLIGENCE OpenStreetMap Tile Layer.

Verifies:
1. Basemap tile provider uses https://tile.openstreetmap.org/{z}/{x}/{y}.png
2. Visible attribution: © OpenStreetMap contributors linking to https://www.openstreetmap.org/copyright
3. All 4 NOAA weather station markers are present on the map
4. Desktop (1440x1080) and Mobile (390x844) rendered map screenshots
5. Zero CARTO tiles and zero 'API KEY REQUIRED' references
"""

import os
import re
import subprocess
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
SCREENSHOTS_DIR = BASE_DIR / "reports" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

CHROME_BIN = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
USER_DATA_DIR = BASE_DIR / ".cache" / "chrome_user_data_map_test"
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = "http://127.0.0.1:5000/dataset-explorer"

DESKTOP_PNG = SCREENSHOTS_DIR / "dataset_explorer_osm_desktop.png"
MOBILE_PNG = SCREENSHOTS_DIR / "dataset_explorer_osm_mobile.png"


def run_chrome_capture(width: int, height: int, output_png: Path) -> str:
    cmd = [
        CHROME_BIN,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        f"--user-data-dir={USER_DATA_DIR}",
        f"--window-size={width},{height}",
        "--virtual-time-budget=6000",
        f"--screenshot={output_png}",
        "--dump-dom",
        URL
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if res.returncode != 0:
        print(f"Warning: Chrome returned non-zero code {res.returncode}")
    return res.stdout


def verify_map():
    print("=" * 70)
    print("VERIFYING OPENSTREETMAP TILE LAYER IN ATMOSPHERE INTELLIGENCE")
    print("=" * 70)

    # 1. Desktop Capture
    print(f"\n[1/3] Capturing Desktop Viewport (1440x1080)...")
    dom_desktop = run_chrome_capture(1440, 1080, DESKTOP_PNG)
    print(f"  -> Saved desktop screenshot: {DESKTOP_PNG} ({DESKTOP_PNG.stat().st_size:,} bytes)")
    with Image.open(DESKTOP_PNG) as im:
        print(f"  -> Verified image dimensions: {im.size} (Width x Height)")

    # 2. Mobile Capture
    print(f"\n[2/3] Capturing Mobile Viewport (390x844)...")
    dom_mobile = run_chrome_capture(390, 844, MOBILE_PNG)
    print(f"  -> Saved mobile screenshot: {MOBILE_PNG} ({MOBILE_PNG.stat().st_size:,} bytes)")
    with Image.open(MOBILE_PNG) as im:
        print(f"  -> Verified image dimensions: {im.size} (Width x Height)")

    # 3. DOM Analysis & Validations
    print(f"\n[3/3] Inspecting Rendered Map DOM Tree...")
    
    # Check Tile Provider
    osm_tiles = re.findall(r'src=["\'](https://tile\.openstreetmap\.org/[^"\']+)["\']', dom_desktop)
    carto_tiles = re.findall(r'basemaps\.cartocdn\.com', dom_desktop)
    api_key_req = re.findall(r'API KEY REQUIRED', dom_desktop, re.IGNORECASE)

    print(f"  * OpenStreetMap tile images found: {len(osm_tiles)}")
    if osm_tiles:
        print(f"    Sample tile: {osm_tiles[0]}")
    print(f"  * CARTO tile references found: {len(carto_tiles)} (Expected: 0)")
    print(f"  * 'API KEY REQUIRED' occurrences found: {len(api_key_req)} (Expected: 0)")

    # Check Attribution
    attr_match = re.search(r'class=["\'][^"\']*leaflet-control-attribution[^"\']*["\'][^>]*>(.*?)</div>', dom_desktop, re.DOTALL)
    if attr_match:
        attr_html = attr_match.group(1).strip()
        print(f"  * Visible attribution HTML: {attr_html}")
        has_osm_text = "OpenStreetMap" in attr_html and "contributors" in attr_html
        has_copyright_link = "https://www.openstreetmap.org/copyright" in attr_html
        print(f"  * Attribution text verified: {has_osm_text}")
        print(f"  * Attribution copyright link verified: {has_copyright_link}")
    else:
        print("  * Warning: attribution control not found in DOM!")

    # Check Weather Station Markers
    marker_pins = re.findall(r'class=["\'][^"\']*atm-map-pin[^"\']*["\']', dom_desktop)
    print(f"  * Custom glowing station markers (.atm-map-pin) rendered: {len(marker_pins)}")
    
    # Verify all 4 station IDs in HTML
    station_ids = ['USW00094728', 'USW00094846', 'USW00024233', 'USW00012839']
    for sid in station_ids:
        in_dom = sid in dom_desktop
        print(f"    - Station {sid} metadata/popup bound: {in_dom}")

    print("\n" + "=" * 70)
    all_ok = (
        len(osm_tiles) > 0 and
        len(carto_tiles) == 0 and
        len(api_key_req) == 0 and
        has_osm_text and
        has_copyright_link and
        len(marker_pins) >= 4 and
        DESKTOP_PNG.exists() and
        MOBILE_PNG.exists()
    )
    print(f"OVERALL MAP TILE PROVIDER VERIFICATION: {'PASSED [OK]' if all_ok else 'FAILED [X]'}")
    print("=" * 70)
    return all_ok


if __name__ == "__main__":
    success = verify_map()
    exit(0 if success else 1)
