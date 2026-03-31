import os
import sys
import time
import subprocess
import numpy as np
import requests
import csv
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional
from collections import deque
from pathlib import Path
import signal
from logging.handlers import RotatingFileHandler

VERSION = "4.4"

DEFAULT_CITIES = [ ... ]   # (same as before - keep your cities list)

# ... (keep the entire ERM_Live_Adaptive class, fetch functions, setup_logging exactly as you had)

def git_backup(data_dir: Path, repo_path: Path):
    """Auto-commit using GitHub Actions built-in token (no PAT needed)"""
    try:
        os.chdir(repo_path)
        subprocess.run(["git", "config", "--global", "user.name", "ERM Bot"], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "erm-bot@github.com"], check=True)
        subprocess.run(["git", "add", str(data_dir)], check=True)
        result = subprocess.run(["git", "commit", "-m", f"ERM auto-backup {datetime.now().isoformat()}"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            subprocess.run(["git", "push"], check=True)
            logging.info("✅ GitHub backup successful")
        else:
            logging.info("No new data to commit this cycle")
    except Exception as e:
        logging.warning(f"Git backup skipped (non-fatal): {e}")

# ... (keep the rest of run_background_worker() exactly as before)

if __name__ == "__main__":
    logging.info(f"ERM Background Worker v{VERSION} started")
    run_background_worker()
