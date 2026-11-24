"""Automated University Submission Packaging Script.

Part of INT396 Unsupervised Machine Learning Project (SIH26073).
Creates clean, professional ZIP archive excluding cache, virtualenvs, and temp files.
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT.parent
TARGET_DIR = PROJECT_ROOT
ZIP_NAME = BASE_DIR / "Faulty_Weather_Station_Detection_SIH26073_Submission.zip"

EXCLUDE_DIRS = {
    ".venv", "venv", ".git", "__pycache__", ".pytest_cache", ".cache", ".ipynb_checkpoints"
}
EXCLUDE_EXTS = {".pyc", ".pyo", ".pyd"}


def create_submission_zip():
    print(f"Packaging {TARGET_DIR.name} into {ZIP_NAME.name} ...")
    
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zip_out:
        for root, dirs, files in os.walk(TARGET_DIR):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in EXCLUDE_EXTS or file.endswith("~"):
                    continue
                    
                file_path = Path(root) / file
                rel_path = file_path.relative_to(BASE_DIR)
                zip_out.write(file_path, arcname=str(rel_path))
                
    size_mb = ZIP_NAME.stat().st_size / (1024 * 1024)
    print(f"Archive created successfully: {ZIP_NAME} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    create_submission_zip()

