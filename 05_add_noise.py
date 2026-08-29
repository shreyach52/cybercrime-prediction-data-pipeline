"""
STEP 6 — ADD NOISE / INCOMPLETENESS

Real NCRP data is messy. If our synthetic data is perfectly clean,
two things go wrong:
  1. Judges who've seen real cybercrime data will immediately clock
     it as fake.
  2. B/C's models will look artificially good during the hackathon,
     then fall apart on any real data later — because they never
     learned to handle missing/messy input.

We add FOUR kinds of realistic mess. Each is applied AFTER Steps 3-5
(i.e. this runs on already-generated complaints/transactions), and
each is controlled by a small probability so you can dial it up/down.
"""

import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# alternate/misspelled versions of district names, as real users would type them
DISTRICT_SPELLING_VARIANTS = {
    "Bengaluru Urban": ["Bengaluru", "Bangalore", "bengaluru urban", "Bengaluru Rural"],
    "Mumbai": ["mumbai", "Bombay", "Mumbai City"],
    "New Delhi": ["Delhi", "delhi", "New delhi"],
    "Pune": ["pune", "Poona"],
}


def add_missing_values(complaints_df: pd.DataFrame, missing_rate: float = 0.06) -> pd.DataFrame:
    """Randomly blank out a small % of non-critical fields — payment_mode
    and victim_district — simulating incomplete form submissions.
    We deliberately do NOT blank fraud_subcategory or amount_lost —
    those are almost always present even in messy real complaints."""
    df = complaints_df.copy()

    for col in ["payment_mode", "victim_district"]:
        mask = np.random.random(len(df)) < missing_rate
        df.loc[mask, col] = None

    return df


def add_duplicate_complaints(complaints_df: pd.DataFrame, duplicate_rate: float = 0.03) -> pd.DataFrame:
    """Simulate victims who file twice (once on the portal, once via the
    1930 helpline, say) — same underlying incident, new complaint_id,
    filed a bit later, sometimes with a slightly different amount typed
    in (people misremember exact figures)."""
    df = complaints_df.copy()
    n_dupes = int(len(df) * duplicate_rate)
    to_duplicate = df.sample(n=n_dupes, random_state=42)

    dupe_rows = []
    for _, row in to_duplicate.iterrows():
        new_row = row.copy()
        new_row["complaint_id"] = f"{row.complaint_id}-DUP"
        new_row["filed_at"] = row.filed_at + pd.Timedelta(hours=random.randint(1, 48))
        if row.amount_lost and row.amount_lost > 0:
            new_row["amount_lost"] = round(row.amount_lost * random.uniform(0.97, 1.03), 2)
        dupe_rows.append(new_row)

    return pd.concat([df, pd.DataFrame(dupe_rows)], ignore_index=True)


def add_background_transactions(transactions_df: pd.DataFrame, accounts_df: pd.DataFrame,
                                 noise_fraction: float = 0.15) -> pd.DataFrame:
    """Add legitimate, non-fraud transactions between random clean accounts
    with NO complaint_id attached. Without this, every single edge in the
    transaction graph is fraud-related, which is unrealistic and makes
    graph detection trivially easy. Real banking graphs are mostly
    legitimate traffic with fraud hidden inside."""
    df = transactions_df.copy()
    clean_accounts = accounts_df.loc[~accounts_df.is_mule_account, "account_id"].tolist()

    n_noise = int(len(df) * noise_fraction)
    base_time = df.timestamp.min()
    time_span = (df.timestamp.max() - df.timestamp.min())

    noise_rows = []
    for i in range(n_noise):
        a, b = random.sample(clean_accounts, 2)
        noise_rows.append({
            "transaction_id": f"TXN-BG-{i+1:06d}",
            "complaint_id": None,          # <-- key marker: not linked to any fraud
            "from_account_id": a,
            "to_account_id": b,
            "amount": round(np.random.lognormal(mean=7.5, sigma=1.0), 2),  # everyday amounts
            "timestamp": base_time + pd.Timedelta(seconds=random.randint(0, int(time_span.total_seconds()))),
            "hop_number": 0,               # 0 = not part of any fraud chain
        })

    return pd.concat([df, pd.DataFrame(noise_rows)], ignore_index=True)


def add_spelling_inconsistency(complaints_df: pd.DataFrame, variant_rate: float = 0.20) -> pd.DataFrame:
    """For districts that have known common misspellings/variants, swap
    in an alternate spelling for a chunk of rows — simulating free-text
    entry inconsistency in the real portal."""
    df = complaints_df.copy()

    for canonical, variants in DISTRICT_SPELLING_VARIANTS.items():
        mask = (df["victim_district"] == canonical) & (np.random.random(len(df)) < variant_rate)
        df.loc[mask, "victim_district"] = [random.choice(variants) for _ in range(mask.sum())]

    return df


def apply_all_noise(complaints_df, transactions_df, accounts_df):
    complaints_df = add_missing_values(complaints_df)
    complaints_df = add_duplicate_complaints(complaints_df)
    complaints_df = add_spelling_inconsistency(complaints_df)
    transactions_df = add_background_transactions(transactions_df, accounts_df)
    return complaints_df, transactions_df


if __name__ == "__main__":
    from importlib import import_module
    step3 = import_module("02_generate_complaints")
    step4 = import_module("03_generate_accounts_atms")
    step5 = import_module("04_generate_transactions_withdrawals")

    complaints = step3.generate_complaints(n=200)
    accounts = step4.generate_accounts(n=300)
    atms = step4.generate_atms(n=150)
    complaints, transactions, withdrawals = step5.generate_transactions_and_withdrawals(
        complaints, accounts, atms
    )

    before_n = len(complaints)
    before_txn = len(transactions)

    complaints, transactions = apply_all_noise(complaints, transactions, accounts)

    print(f"Complaints: {before_n} -> {len(complaints)} after adding duplicates")
    print(f"Missing payment_mode: {complaints.payment_mode.isna().sum()}")
    print(f"Missing victim_district: {complaints.victim_district.isna().sum()}")
    print(f"Duplicate complaints added: {complaints.complaint_id.str.contains('-DUP').sum()}")
    print()
    print(f"Transactions: {before_txn} -> {len(transactions)} after adding background noise")
    print(f"Background (non-fraud) transactions: {transactions.complaint_id.isna().sum()}")
    print(f"  -> that's {transactions.complaint_id.isna().mean()*100:.1f}% of all transactions with NO fraud link")
    print()
    print("Sample of district spelling variants now present:")
    print(complaints["victim_district"].value_counts().head(15))