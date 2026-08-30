# Synthetic Cybercrime Complaint Data Pipeline

**Smart India Hackathon 2026 — PS 26184 (Ministry of Home Affairs / I4C)**


## The problem

India's National Cybercrime Reporting Portal (NCRP) receives ~8,000
complaints daily. The official problem statement asks for a predictive
system that forecasts **where and when fraud money will be withdrawn in
cash**, so law enforcement and banks can intervene before it disappears —
instead of just reacting after the fact.

Our team built a 5-part system on top of this idea (graph-based mule
network detection, spatio-temporal hotspot prediction, a risk-fusion +
alerting engine, and a GIS dashboard). **This repo is my part**: the data
foundation everyone else's models were built and tested against.

## What this pipeline does

Real transaction-level cybercrime data isn't publicly available, so I
built a synthetic data generator calibrated against real published
NCRP/I4C/MHA statistics — not arbitrary random data. It simulates the
full lifecycle of a cybercrime complaint: money leaving a victim's
account, bouncing through a chain of mule accounts, and finally being
withdrawn in cash at an ATM — with a live-feed API to demo it in real
time.

**Key engineering decisions, not just "generate some fake rows":**

- **Calibrated, not arbitrary** — fraud-type frequency and loss
  distributions are matched to real I4C figures (e.g. investment scams
  are ~12% of complaints but ~77% of ₹ losses; the generator models
  volume and value as *two separate* weighted distributions to capture
  that).
- **Engineered reuse patterns** — ~15-20% of accounts and ATMs are
  seeded as a "hot pool" that gets pulled disproportionately often,
  simulating real mule-network and cash-out-hotspot behavior — the
  actual signal a graph/network model needs to learn to detect.
- **Ground-truth separation** — every prediction target (the eventual
  cash-out location/time, whether an account is a mule) is generated
  but stored *separately* from the visible features, so downstream ML
  models can't accidentally "see the answer" during training. This
  mirrors how a real evaluation pipeline has to work.
- **Deliberately imperfect** — missing fields, duplicate complaints,
  inconsistent spellings, and background non-fraud transactions are
  injected on purpose, so the dataset doesn't look artificially clean.
- **Automated validation** — a test suite checks every engineered
  distribution against its target range (13/13 checks passing) before
  the data is considered ready to hand off.
- **Train/test split with held-back outcomes** — 80% of complaints
  include their full resolved history (for model training); the
  remaining 20% simulate *live, unresolved* complaints with the
  eventual withdrawal deliberately withheld into a separate ground-truth
  file, used only for scoring prediction accuracy.

## Pipeline stages

| Stage | Script | Output |
|---|---|---|
| Research & calibration | — | Real NCRP/I4C stats used as generation targets |
| Schema design | `01_schema.py` | 5 core entities (Complaint, Account, ATM, Transaction, Withdrawal) |
| Complaint generation | `02_generate_complaints.py` | Complaints matching real fraud-type/loss distributions |
| Account/ATM generation | `03_generate_accounts_atms.py` | Entities with a weighted mule/hotspot reuse pool |
| Transaction chains | `04_generate_transactions_withdrawals.py` | Fraud chains linking complaints → accounts → ATMs |
| Noise injection | `05_add_noise.py` | Missing data, duplicates, spelling inconsistency, background noise |
| Validation | `06_validate.py` | Automated distribution checks |
| Export | `07_export.py` | Final CSV/JSON dataset with train/test split |
| Ingestion API | `08_ingestion_api.py` | FastAPI service simulating a live complaint feed |

## Tech stack

Python, pandas, NumPy, Faker, FastAPI, Uvicorn

## Running it

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

python 07_export.py                              # generates data/
python -m uvicorn 08_ingestion_api:app --reload   # starts the live API
```

Then visit `http://127.0.0.1:8000/docs` for an interactive API explorer —
try `POST /simulate/start-feed` to watch complaints arrive on a live
timer, exactly as they would from a real ingestion pipeline.

## Sample validation output

```
── Loss concentration (should mirror real I4C data) ───
  [PASS] Investment Scam share of TOTAL losses: 71.9%  (expected 55-85%)

── Account / ATM reuse (mule network signal) ──────────
  [PASS] Mule account pool size: 18.0%  (expected 14-22%)
  [PASS] Actual mule-account hit rate in fraud txns: 77.3%  (expected 60-90%)

── Ground truth integrity ──────────────────────────────
  [PASS] Non-financial complaints WITHOUT ground truth (leak check): 0.0%  (expected 0-0%)

RESULT: 13/13 checks passed
```

## Scope note

This repo contains only my individual contribution (data generation +
ingestion API) from a 5-member SIH team project. The graph/network
model, spatio-temporal model, fusion/alerting backend, and GIS dashboard
were built by teammates and aren't included here.
