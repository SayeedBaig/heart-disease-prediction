from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Upload directories
UPLOAD_DIR = PROJECT_ROOT / "uploads"
ECG_UPLOAD_DIR = UPLOAD_DIR / "ecg"
ECHO_UPLOAD_DIR = UPLOAD_DIR / "echo"


# Allowed ECG file formats
ALLOWED_ECG_EXTENSIONS = {
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# Allowed Echo file formats
ALLOWED_ECHO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}