"""
STEP 9 — INGESTION API

Two jobs:
  1. Serve the exported dataset over HTTP (GET endpoints) so B/C/D
     don't need to touch CSV files directly — they just hit an endpoint.
  2. Simulate a LIVE feed: POST /simulate/start-feed replays the held-out
     "test" complaints (data/complaints_test.csv) one at a time on a
     timer, appending each to an in-memory list. GET /complaints shows
     that list growing over time — this is what makes the demo look
     like a real, live NCRP stream instead of a static file dump.

HOW TO RUN:
  uvicorn 08_ingestion_api:app --reload

HOW TO TEST (no extra tools needed):
  Open http://127.0.0.1:8000/docs in a browser — FastAPI auto-generates
  an interactive test page. Click any endpoint, "Try it out", "Execute".
"""

import json
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="PS 26184 — Cybercrime Complaint Ingestion API")

DATA_DIR = Path("data")

# ── in-memory "live" complaint log ──────────────────────────────────
# In a real system this would be a database. For the hackathon, a
# plain Python list is fine — it resets every time you restart the
# server, which is actually convenient for repeated demo runs.
live_complaints = []
feed_status = {"running": False, "sent": 0, "total": 0}


class Complaint(BaseModel):
    """Defines the shape of a complaint POSTed in from outside —
    FastAPI uses this to auto-validate incoming JSON and reject
    anything malformed, without you writing any validation code."""
    complaint_id: str
    filed_at: str
    fraud_category: str
    fraud_subcategory: str
    amount_lost: float
    victim_district: Optional[str] = None
    victim_state: Optional[str] = None
    payment_mode: Optional[str] = None
    reported_delay_hours: float


def _load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name}.csv not found — run 07_export.py first")
    return pd.read_csv(path)


# ── bulk endpoints: serve the static exported dataset ───────────────

@app.get("/")
def root():
    return {"message": "PS 26184 ingestion API is running. See /docs for interactive testing."}


@app.get("/complaints/historical")
def get_historical_complaints(limit: int = 100):
    """Training data — full history including known outcomes."""
    df = _load_csv("complaints_train")
    return df.head(limit).to_dict(orient="records")


@app.get("/accounts")
def get_accounts(limit: int = 200):
    df = _load_csv("accounts_visible")
    return df.head(limit).to_dict(orient="records")


@app.get("/atms")
def get_atms(limit: int = 200):
    df = _load_csv("atms_visible")
    return df.head(limit).to_dict(orient="records")


@app.get("/transactions")
def get_transactions(complaint_id: Optional[str] = None, limit: int = 200):
    df = _load_csv("transactions")
    if complaint_id:
        df = df[df.complaint_id == complaint_id]
    return df.head(limit).to_dict(orient="records")


# ── live feed: this is the "simulated NCRP stream" ──────────────────

@app.post("/complaints")
def ingest_complaint(complaint: Complaint):
    """External systems (or our own simulator below) POST a new
    complaint here, exactly like a real NCRP feed pushing in a fresh
    report. Appends to the in-memory live log."""
    live_complaints.append(complaint.dict())
    return {"status": "received", "complaint_id": complaint.complaint_id, "total_live": len(live_complaints)}


@app.get("/complaints")
def get_live_complaints():
    """What the dashboard polls to see new complaints arrive."""
    return {"count": len(live_complaints), "complaints": live_complaints}


@app.get("/complaints/{complaint_id}")
def get_one_complaint(complaint_id: str):
    for c in live_complaints:
        if c["complaint_id"] == complaint_id:
            return c
    raise HTTPException(status_code=404, detail="Not found in live feed yet")


def _run_feed(interval_seconds: float, count: int):
    """Runs in a background thread: reads the held-out test complaints
    and POSTs (well, directly appends) one every `interval_seconds`,
    simulating complaints arriving over time instead of all at once."""
    df = _load_csv("complaints_test").head(count)
    feed_status["running"] = True
    feed_status["total"] = len(df)
    feed_status["sent"] = 0

    for _, row in df.iterrows():
        record = json.loads(row.to_json())
        live_complaints.append(record)
        feed_status["sent"] += 1
        time.sleep(interval_seconds)

    feed_status["running"] = False


@app.post("/simulate/start-feed")
def start_feed(interval_seconds: float = 2.0, count: int = 30):
    """Kicks off the live-feed simulation in the background so this
    request returns immediately — the feed keeps running afterward.
    Watch it happen by polling GET /complaints or GET /simulate/status."""
    if feed_status["running"]:
        return {"status": "already running", "progress": feed_status}

    thread = threading.Thread(target=_run_feed, args=(interval_seconds, count), daemon=True)
    thread.start()
    return {"status": "started", "interval_seconds": interval_seconds, "count": count}


@app.get("/simulate/status")
def simulate_status():
    return feed_status


@app.post("/simulate/reset")
def reset_feed():
    """Clears the live log — handy between demo runs/practice."""
    live_complaints.clear()
    feed_status.update({"running": False, "sent": 0, "total": 0})
    return {"status": "reset"}