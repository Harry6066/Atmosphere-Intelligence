"""Submission Packaging Script for ATMOSPHERE INTELLIGENCE (SIH26073).

Creates a clean, reproducible ZIP submission archive.
Excludes virtual environments, bytecode caches, temp directories, and secrets.
"""

import os
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
ZIP_NAME = "SIH26073_ATMOSPHERE_INTELLIGENCE_SUBMISSION.zip"
TARGET_ZIP = WORKSPACE_DIR / ZIP_NAME
REPORTS_ZIP = BASE_DIR / "reports" / ZIP_NAME

EXCLUDE_DIRS = {
    ".venv", "venv", "env", ".python", ".uv_cache", ".uv_python",
    "__pycache__", ".pytest_cache", ".cache", ".git", ".idea",
    ".vscode", ".ipynb_checkpoints", ".streamlit"
}

EXCLUDE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".log", ".tmp", ".bak", ".swp", ".zip"
}


def should_include(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    for part in parts:
        if part in EXCLUDE_DIRS:
            return False
        if "chrome_user_data" in part or ".matplotlib" in part:
            return False
    ext = os.path.splitext(rel_path)[1].lower()
    if ext in EXCLUDE_EXTENSIONS:
        return False
    return True


def create_submission_zip():
    print(f"Creating submission archive: {TARGET_ZIP}")
    file_count = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(TARGET_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(BASE_DIR):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and "chrome_user_data" not in d and not d.startswith(".cache")]
            
            for file in files:
                abs_file = Path(root) / file
                rel_file = abs_file.relative_to(BASE_DIR)
                rel_str = str(rel_file).replace("\\", "/")
                
                # Check exclusion filter
                if not should_include(rel_str):
                    continue
                
                # Don't zip our own packaging script or existing zip files
                if file == ZIP_NAME or file.endswith(".zip"):
                    continue

                arcname = f"Faulty_Weather_Station_Detection/{rel_str}"
                zf.write(abs_file, arcname=arcname)
                file_count += 1
                total_uncompressed_bytes += abs_file.stat().st_size

    # Copy to reports as well
    import shutil
    shutil.copyfile(TARGET_ZIP, REPORTS_ZIP)

    zip_size_bytes = TARGET_ZIP.stat().st_size
    zip_size_mb = zip_size_bytes / (1024 * 1024)
    raw_size_mb = total_uncompressed_bytes / (1024 * 1024)

    print("\n" + "=" * 70)
    print("SUBMISSION ZIP PACKAGE GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Archive Name:            {ZIP_NAME}")
    print(f"Primary Location:        {TARGET_ZIP}")
    print(f"Backup in reports/:      {REPORTS_ZIP}")
    print(f"Archived Files Count:    {file_count} files")
    print(f"Uncompressed Data Size:  {raw_size_mb:.2f} MB ({total_uncompressed_bytes:,} bytes)")
    print(f"Compressed Archive Size: {zip_size_mb:.2f} MB ({zip_size_bytes:,} bytes)")
    print(f"Compression Ratio:       {(1 - zip_size_bytes / total_uncompressed_bytes) * 100:.1f}% space savings")
    print("=" * 70)


if __name__ == "__main__":
    create_submission_zip()

