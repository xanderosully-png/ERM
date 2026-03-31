from fastapi import FastAPI
import uvicorn
import os
import subprocess
from datetime import datetime
import sys
import numpy as np
import requests
import csv
import logging
from pathlib import Path
from collections import deque
import json
from typing import List, Dict, Optional

app = FastAPI(title="ERM Live Update Service")

VERSION = "4.4"

DEFAULT_CITIES = [ ... ]  # (copy your full DEFAULT_CITIES list from erm_background_worker.py here)

class ERM_Live_Adaptive:
    # (copy the entire ERM_Live_Adaptive class from your background worker here)
    pass   # ← paste the full class here

# (copy fetch_multi_variable_data, load_cities, and the step/predict logic exactly as in your background worker)

@app.get("/update")
async def update_data():
    """Ping this endpoint to collect new data and save to GitHub"""
    try:
        # Run the same collection logic as the background worker
        base_dir = Path(__file__).parent
        data_dir = base_dir / "ERM_Data"
        data_dir.mkdir(parents=True, exist_ok=True)

        cities = load_cities(base_dir)  # or DEFAULT_CITIES
        # ... (run the full collection loop exactly as in your background worker)

        # Commit and push
        subprocess.run(["git", "config", "--global", "user.name", "ERM Bot"], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "erm-bot@github.com"], check=True)
        subprocess.run(["git", "add", str(data_dir)], check=True)
        subprocess.run(["git", "commit", "-m", f"ERM live update {datetime.now().isoformat()}"], check=True)
        subprocess.run(["git", "push"], check=True)

        return {"status": "success", "message": f"Updated {len(cities)} cities at {datetime.now()}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health():
    return {"status": "healthy", "time": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
