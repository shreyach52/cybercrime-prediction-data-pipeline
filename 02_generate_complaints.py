"""
STEP 3 — GENERATE COMPLAINTS

Calibration source (from our research step): investment scams are
~12% of complaint volume but ~77% of rupee losses. Small-value scams
(UPI, OTP-based, e-wallet) are the opposite — high volume, low value
per incident. We model that by giving each fraud subcategory its
OWN pair of numbers: a frequency weight and a lognormal amount
distribution (mean/sigma), instead of one shared distribution.

Design note: only "Financial Fraud" complaints go on to generate a
transaction chain / withdrawal later — a complaint about stalking or
hacking has no cash-out to predict. We simulate that split here too,
since it's realistic and it means the final dataset correctly has
some complaints with NO linked withdrawal (nothing to predict) —
which matters, otherwise your model just learns "every complaint has
a cash-out," which is false in real NCRP data.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

fake = Faker("en_IN")
random.seed(42)      # fixes the "random" sequence so results are reproducible —
np.random.seed(42)   # remove/change this later if you want fresh data each run

# ── Fraud subcategory table ──────────────────────────────────────────
# freq_weight   → relative share of COMPLAINT COUNT (drives how often sampled)
# amt_mean/sigma → parameters of the lognormal distribution for amount_lost
#                  (higher mean = bigger typical loss for that subtype)
FRAUD_SUBCATEGORIES = [
    # name                              category            freq_weight  amt_mean  amt_sigma
    ("UPI Related Frauds",              "Financial Fraud",  35,          8.0,      0.9),
    ("Debit/Credit Card Fraud",         "Financial Fraud",  15,          8.5,      1.0),
    ("Internet Banking Related Fraud",  "Financial Fraud",  10,          8.8,      1.0),
    ("E-Wallet Related Fraud",          "Financial Fraud",   8,          7.6,      0.8),
    ("Investment Scam",                 "Financial Fraud",  12,         11.5,      1.3),  # big amounts
    ("Digital Arrest / Impersonation",  "Financial Fraud",   5,         11.0,      1.2),  # big amounts
    ("Online Job Fraud",                "Financial Fraud",   5,          8.2,      0.9),
    ("Cyber Bullying/Stalking",         "Other Cyber Crime", 4,          0,        0),    # no money lost
    ("Hacking/Damage to Computer",      "Other Cyber Crime", 3,          0,        0),    # no money lost
    ("Online Matrimonial/Romance Scam", "Other Cyber Crime", 3,          9.5,      1.1),
]

# Karnataka-weighted but nationally spread — Maharashtra/UP lead nationally,
# Karnataka included at realistic ~12% share per the IndiaSpend/I4C figures
DISTRICTS_WITH_STATE = [
    ("Bengaluru Urban", "Karnataka", 12),
    ("Mumbai", "Maharashtra", 10),
    ("Pune", "Maharashtra", 8),
    ("Lucknow", "Uttar Pradesh", 7),
    ("Noida", "Uttar Pradesh", 7),
    ("Ahmedabad", "Gujarat", 6),
    ("New Delhi", "Delhi", 6),
    ("Hyderabad", "Telangana", 5),
    ("Chennai", "Tamil Nadu", 5),
    ("Jaipur", "Rajasthan", 4),
]

PAYMENT_MODES_WEIGHTED = [("UPI", 45), ("Card", 20), ("Net Banking", 20), ("Wallet", 15)]


def _weighted_pick(options_with_weights):
    """options_with_weights: list of (value, weight) tuples → returns one value."""
    values = [o[0] for o in options_with_weights]
    weights = [o[-1] for o in options_with_weights]
    return random.choices(values, weights=weights, k=1)[0]


def generate_complaints(n: int, start_date: str = "2026-06-01", end_date: str = "2026-08-29") -> pd.DataFrame:
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    span_seconds = int((end - start).total_seconds())

    rows = []
    for i in range(n):
        # pick subcategory using ONLY the frequency weight
        sub = random.choices(
            FRAUD_SUBCATEGORIES,
            weights=[s[2] for s in FRAUD_SUBCATEGORIES],
            k=1,
        )[0]
        subcategory, category, _freq_w, amt_mean, amt_sigma = sub

        # amount_lost: 0 for non-financial complaints, else sampled from
        # THAT subcategory's own lognormal curve
        amount_lost = 0.0 if amt_mean == 0 else round(np.random.lognormal(amt_mean, amt_sigma), 2)

        chosen = random.choices(DISTRICTS_WITH_STATE, weights=[d[2] for d in DISTRICTS_WITH_STATE], k=1)[0]
        district, state, _ = chosen

        filed_at = start + timedelta(seconds=random.randint(0, span_seconds))

        # reporting delay: most people report within a day, long tail out to weeks —
        # another lognormal, small mean so most delays are small
        reported_delay_hours = round(np.random.lognormal(mean=1.5, sigma=1.2), 2)

        rows.append({
            "complaint_id": f"CMP-{i+1:06d}",
            "filed_at": filed_at,
            "fraud_category": category,
            "fraud_subcategory": subcategory,
            "amount_lost": amount_lost,
            "victim_district": district,
            "victim_state": state,
            "payment_mode": _weighted_pick(PAYMENT_MODES_WEIGHTED) if amount_lost > 0 else None,
            "reported_delay_hours": reported_delay_hours,
            "victim_account_id": None,     # filled in Step 4 (accounts)
            "suspect_account_id": None,    # filled in Step 4
            "true_cashout_atm_id": None,   # filled in Step 5 (withdrawals)
            "true_cashout_window_start": None,
            "true_cashout_window_end": None,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_complaints(n=20)
    print(df[["complaint_id", "filed_at", "fraud_category", "fraud_subcategory", "amount_lost", "victim_district", "reported_delay_hours"]])
    print("\nfraud_subcategory value counts (out of 20 — small sample, won't match weights exactly):")
    print(df["fraud_subcategory"].value_counts())
    print("\nTotal amount lost by subcategory (should show Investment Scam dominating despite fewer rows):")
    print(df.groupby("fraud_subcategory")["amount_lost"].sum().sort_values(ascending=False))
