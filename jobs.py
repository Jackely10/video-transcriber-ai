from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Nur für lokale Entwicklung - Railway braucht das nicht
from dotenv import load_dotenv
if not os.environ.get("RAILWAY_ENVIRONMENT"):
    load_dotenv()

DEFAULT_SUMMARY_LANG = os.getenv("SUMMARY_UI_LANG", "auto")
