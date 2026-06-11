"""Hardcoded runtime configuration for the Raspberry Pi client."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

# Backend contract
BACKEND_URL = "https://final-year-backend.fly.dev"
DEVICE_ID = "PI-001"

# Polling and upload cadence
CONFIG_REFRESH_INTERVAL = 10
EMBEDDING_REFRESH_INTERVAL = 300
UPLOAD_INTERVAL = 60
API_TIMEOUT_SECONDS = 10
API_MAX_RETRIES = 3
API_RETRY_BACKOFF_SECONDS = 2

# Recognition behavior
CONFIDENCE_THRESHOLD = 0.60
TRACK_EXPIRY_SECONDS = 5
CAMERA_INDEX = 0
FACE_DET_SIZE = (640, 640)
RECOGNITION_FRAME_DELAY_SECONDS = 0.3

# Local storage
STORAGE_DIR = REPO_ROOT / "storage"
PENDING_BATCHES_FILE = STORAGE_DIR / "pending_batches.json"
STATE_FILE = STORAGE_DIR / "state.json"
ATTENDANCE_EXPORT_DIR = REPO_ROOT / "attendance_exports"
LOG_DIR = REPO_ROOT / "logs"
LOG_FILE = LOG_DIR / "pi.log"

# InsightFace model cache location.
INSIGHTFACE_MODEL_ROOT = str(REPO_ROOT / "models")

# Logging
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5
