"""
STEP 4 — GENERATE ACCOUNTS + ATMs (with the reuse pool mechanic)

Key idea: reuse must be built INTO the sampling, not injected after
the fact (see our earlier fix to the original workflow). We do this
with a "hot pool" pattern:

  - Generate a full list of accounts/ATMs as normal.
  - Mark a SMALL SUBSET (15-20%) as "hot" (is_mule_account=True /
    is_known_hotspot=True).
  - Whenever some other part of the pipeline (Step 5: Transactions/
    Withdrawals) needs to PICK an account or ATM to use, it doesn't
    pick uniformly at random from everyone — it rolls a weighted dice:
    ~70-80% chance of picking from the "hot" pool, ~20-30% chance of
    picking a fresh/clean one.

  That's what produces realistic reuse: the same 15-20% of accounts/
  ATMs show up over and over across many different complaints, which
  is exactly the pattern B's graph community-detection algorithm
  (Louvain) and centrality scoring (PageRank) are designed to catch.

We also add the device_fingerprint reuse signal here (from the
schema) — mule accounts in the hot pool disproportionately share a
small set of fingerprints too, a second independent signal.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

fake = Faker("en_IN")
random.seed(42)
np.random.seed(42)

BANKS = ["State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank",
         "Punjab National Bank", "Bank of Baroda", "Kotak Mahindra Bank"]

DISTRICTS = ["Bengaluru Urban", "Mumbai", "Pune", "Lucknow", "Noida",
             "Ahmedabad", "New Delhi", "Hyderabad", "Chennai", "Jaipur"]

# rough real-world bounding boxes (lat, lon) per district, for placing ATMs
# realistically instead of pure-random coordinates scattered anywhere
DISTRICT_COORDS = {
    "Bengaluru Urban": (12.97, 77.59),
    "Mumbai": (19.07, 72.87),
    "Pune": (18.52, 73.86),
    "Lucknow": (26.85, 80.95),
    "Noida": (28.54, 77.39),
    "Ahmedabad": (23.02, 72.57),
    "New Delhi": (28.61, 77.21),
    "Hyderabad": (17.39, 78.49),
    "Chennai": (13.08, 80.27),
    "Jaipur": (26.91, 75.79),
}


def generate_accounts(n: int, mule_pool_fraction: float = 0.18) -> pd.DataFrame:
    """mule_pool_fraction: what fraction of accounts are 'hot' mule accounts (15-20% target)."""
    n_mule = int(n * mule_pool_fraction)

    # a small number of shared device fingerprints for the mule pool to reuse —
    # far fewer fingerprints than mule accounts, so reuse is forced
    shared_fingerprints = [fake.sha1()[:12] for _ in range(max(3, n_mule // 8))]

    rows = []
    for i in range(n):
        is_mule = i < n_mule  # first n_mule accounts are the hot pool
        rows.append({
            "account_id": f"ACC-{i+1:06d}",
            "account_holder_name": fake.name(),
            "bank_name": random.choice(BANKS),
            "account_type": random.choices(["Savings", "Current"], weights=[85, 15], k=1)[0],
            "opened_at": fake.date_time_between(start_date="-3y", end_date="-30d"),
            "is_mule_account": is_mule,
            "home_district": random.choice(DISTRICTS),
            # mule accounts draw from the small shared fingerprint pool (forced reuse);
            # clean accounts each get their own unique fingerprint
            "device_fingerprint": random.choice(shared_fingerprints) if is_mule else fake.sha1()[:12],
        })

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle so mules aren't all at the top


def generate_atms(n: int, hotspot_pool_fraction: float = 0.17) -> pd.DataFrame:
    n_hot = int(n * hotspot_pool_fraction)

    rows = []
    for i in range(n):
        is_hot = i < n_hot
        district = random.choice(DISTRICTS)
        base_lat, base_lon = DISTRICT_COORDS[district]
        # jitter the coordinates a little so ATMs in the same district aren't
        # all stacked on the exact same point
        lat = round(base_lat + random.uniform(-0.05, 0.05), 5)
        lon = round(base_lon + random.uniform(-0.05, 0.05), 5)

        rows.append({
            "atm_id": f"ATM-{i+1:05d}",
            "bank_name": random.choice(BANKS),
            "district": district,
            "latitude": lat,
            "longitude": lon,
            "is_known_hotspot": is_hot,
        })

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def pick_account(accounts_df: pd.DataFrame, hot_pull_rate: float = 0.75) -> str:
    """Weighted picker used later (Step 5): ~75% chance of pulling from the
    mule pool, ~25% chance of a clean account — this is what actually
    PRODUCES the 15-20% reuse pattern in the transaction data."""
    mule_ids = accounts_df.loc[accounts_df.is_mule_account, "account_id"].tolist()
    clean_ids = accounts_df.loc[~accounts_df.is_mule_account, "account_id"].tolist()
    if random.random() < hot_pull_rate and mule_ids:
        return random.choice(mule_ids)
    return random.choice(clean_ids)


def pick_atm(atms_df: pd.DataFrame, hot_pull_rate: float = 0.75) -> str:
    hot_ids = atms_df.loc[atms_df.is_known_hotspot, "atm_id"].tolist()
    cold_ids = atms_df.loc[~atms_df.is_known_hotspot, "atm_id"].tolist()
    if random.random() < hot_pull_rate and hot_ids:
        return random.choice(hot_ids)
    return random.choice(cold_ids)


if __name__ == "__main__":
    accounts = generate_accounts(n=500)
    atms = generate_atms(n=200)

    print(f"Accounts: {len(accounts)} total, {accounts.is_mule_account.sum()} mule ({accounts.is_mule_account.mean()*100:.1f}%)")
    print(f"ATMs:     {len(atms)} total, {atms.is_known_hotspot.sum()} hotspot ({atms.is_known_hotspot.mean()*100:.1f}%)")
    print()
    print("Sample accounts:")
    print(accounts.head(5))
    print()
    print("Sample ATMs:")
    print(atms.head(5))
    print()

    # PROVE the reuse mechanic works: pick 1000 accounts using pick_account()
    # and check what fraction actually land on mule accounts
    picks = [pick_account(accounts) for _ in range(1000)]
    mule_id_set = set(accounts.loc[accounts.is_mule_account, "account_id"])
    reuse_rate = sum(1 for p in picks if p in mule_id_set) / len(picks)
    print(f"Out of 1000 simulated picks, {reuse_rate*100:.1f}% landed on mule accounts")
    print(f"(this is the number that will show up as 'account reuse' once Step 5 wires this into real transactions)")
