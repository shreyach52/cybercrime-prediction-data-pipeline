"""
STEP 8 — EXPORT

Produces the final files your team actually loads. Key design choice:
an 80/20 TRAIN/TEST split over complaints.

  TRAIN (80%): full history — complaint, transaction chain, AND the
    withdrawal that happened. This is what B/C's models learn from.

  TEST (20%): complaint + transaction chain only. The withdrawal is
    stripped out and held back in a separate ground_truth file. This
    simulates a live, unresolved complaint — exactly the situation
    the real system faces, where the cash-out hasn't happened yet
    and predicting it is the entire point.

is_mule_account / is_known_hotspot are LABELS, not features — they
never appear in the visible accounts/atms files. They go in separate
_labels.csv files used only to grade whether the graph/spatial models
correctly identified risk, never given to the models as input.

Output structure (all in data/):
  complaints_train.csv         - full history, all fields
  complaints_test.csv          - live complaints, cashout fields = null
  complaints_test_ground_truth.csv  - the held-back answer key for test complaints
  accounts_visible.csv         - no is_mule_account column
  accounts_labels.csv          - account_id + is_mule_account only
  atms_visible.csv             - no is_known_hotspot column
  atms_labels.csv              - atm_id + is_known_hotspot only
  transactions.csv             - ALL transactions (train+test+background noise) —
                                  this represents "money movement observed so far"
  withdrawals_train.csv        - withdrawals ONLY for train complaints (historical)
"""

import json
import pandas as pd
import numpy as np
from importlib import import_module
from pathlib import Path

step3 = import_module("02_generate_complaints")
step4 = import_module("03_generate_accounts_atms")
step5 = import_module("04_generate_transactions_withdrawals")
step6 = import_module("05_add_noise")

OUTPUT_DIR = Path("data")


def export_dataset(n_complaints=3000, n_accounts=1200, n_atms=400, test_fraction=0.2, seed=42):
    OUTPUT_DIR.mkdir(exist_ok=True)
    rng = np.random.RandomState(seed)

    print("Generating full pipeline...")
    complaints = step3.generate_complaints(n=n_complaints)
    accounts = step4.generate_accounts(n=n_accounts)
    atms = step4.generate_atms(n=n_atms)
    complaints, transactions, withdrawals = step5.generate_transactions_and_withdrawals(
        complaints, accounts, atms
    )
    complaints, transactions = step6.apply_all_noise(complaints, transactions, accounts)

    # ── train/test split ────────────────────────────────────────────
    complaint_ids = complaints.complaint_id.tolist()
    is_test = rng.random(len(complaint_ids)) < test_fraction
    complaints["_is_test"] = is_test

    complaints_train = complaints[~complaints._is_test].drop(columns=["_is_test"]).reset_index(drop=True)
    complaints_test_full = complaints[complaints._is_test].drop(columns=["_is_test"]).reset_index(drop=True)

    # ground truth file: only for test complaints that actually HAD a cashout
    ground_truth = complaints_test_full.loc[
        complaints_test_full.true_cashout_atm_id.notna(),
        ["complaint_id", "true_cashout_atm_id", "true_cashout_window_start", "true_cashout_window_end"]
    ].copy()
    # also attach the real withdrawal amount/account, pulled from withdrawals table,
    # since E's scorecard needs the full picture to grade against
    wd_lookup = withdrawals.set_index("complaint_id")[["account_id", "amount", "withdrawn_at"]]
    ground_truth = ground_truth.join(wd_lookup, on="complaint_id", rsuffix="_actual")

    # strip cashout fields from the TEST complaints file — this is the "live" view
    complaints_test = complaints_test_full.copy()
    for col in ["true_cashout_atm_id", "true_cashout_window_start", "true_cashout_window_end"]:
        complaints_test[col] = None

    # ── accounts / atms: split visible features from ground-truth labels ──
    accounts_visible = accounts.drop(columns=["is_mule_account"])
    accounts_labels = accounts[["account_id", "is_mule_account"]]
    atms_visible = atms.drop(columns=["is_known_hotspot"])
    atms_labels = atms[["atm_id", "is_known_hotspot"]]

    # ── transactions: everyone gets to see the money movement so far ──
    # (this is fine — the transaction CHAIN is observable in real time;
    # it's only the eventual WITHDRAWAL that's the future unknown event)
    transactions_export = transactions.copy()

    # ── withdrawals: only for TRAIN complaints — test withdrawals are
    # exactly what's being predicted, so they must not appear here ──
    train_ids = set(complaints_train.complaint_id)
    withdrawals_train = withdrawals[withdrawals.complaint_id.isin(train_ids)].reset_index(drop=True)

    # ── write everything out, both CSV and JSON ─────────────────────
    files = {
        "complaints_train": complaints_train,
        "complaints_test": complaints_test,
        "complaints_test_ground_truth": ground_truth,
        "accounts_visible": accounts_visible,
        "accounts_labels": accounts_labels,
        "atms_visible": atms_visible,
        "atms_labels": atms_labels,
        "transactions": transactions_export,
        "withdrawals_train": withdrawals_train,
    }

    for name, df in files.items():
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
        df.to_json(OUTPUT_DIR / f"{name}.json", orient="records", date_format="iso", indent=2)

    print(f"\nExported {len(files)} tables to {OUTPUT_DIR}/ (CSV + JSON each):\n")
    for name, df in files.items():
        print(f"  {name:32s} {len(df):>6d} rows")

    print(f"\nTrain complaints: {len(complaints_train)}  |  Test (live) complaints: {len(complaints_test)}")
    print(f"Test complaints with a real cashout to predict: {len(ground_truth)}")

    return files


if __name__ == "__main__":
    export_dataset()