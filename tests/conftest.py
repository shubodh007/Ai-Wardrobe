import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("API_DISABLE_MODEL_LOAD", "1")
os.environ.setdefault("RATE_LIMIT_ENABLED", "0")
os.environ.setdefault("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024))
