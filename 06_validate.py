"""
STEP 7 — VALIDATE

Runs the full pipeline (Steps 3-6) end to end, then checks every
distribution we deliberately engineered against its target range.
This is the "did I actually build what I meant to build" check —
run this ANY time you tweak a weight/rate in an earlier step, before
telling the team the new data is ready.

Each check prints PASS/FAIL with the actual number vs. the expected
range. A FAIL doesn't necessarily mean something is broken — small
sample sizes cause natural variance — but it tells you where to look.
"""

from importlib import import_module
import pandas as pd

step3 = import_module("02_generate_complaints")
step4 = import_module("03_generate_accounts_atms")
step5 = import_module("04_generate_transactions_withdrawals")
step6 = import_module("05_add_noise")


def check(label, actual, low, high, unit=""):
    passed = low <= actual <= high
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}: {actual}{unit}  (expected {low}-{high}{unit})")
    return passed


def validate(n_complaints=2000, n_accounts=800, n_atms=300):
    print(f"Generating pipeline: {n_complaints} complaints, {n_accounts} accounts, {n_atms} ATMs...\n")

    complaints = step3.generate_complaints(n=n_complaints)
    accounts = step4.generate_accounts(n=n_accounts)
    atms = step4.generate_atms(n=n_atms)
    complaints, transactions, withdrawals = step5.generate_transactions_and_withdrawals(
        complaints, accounts, atms
    )
    complaints, transactions = step6.apply_all_noise(complaints, transactions, accounts)

    results = []

    print("── Complaint volume & category split ──────────────────")
    results.append(check("Total complaints (incl. duplicates)", len(complaints),
                          n_complaints, int(n_complaints * 1.1)))
    upi_share = (complaints.fraud_subcategory == "UPI Related Frauds").mean() * 100
    results.append(check("UPI Fraud complaint share", round(upi_share, 1), 28, 42, "%"))
    investment_share = (complaints.fraud_subcategory == "Investment Scam").mean() * 100
    results.append(check("Investment Scam complaint share", round(investment_share, 1), 8, 18, "%"))

    print("\n── Loss concentration (should mirror real I4C data) ───")
    total_loss = complaints.amount_lost.sum()
    inv_loss_share = complaints.loc[complaints.fraud_subcategory == "Investment Scam", "amount_lost"].sum() / total_loss * 100
    results.append(check("Investment Scam share of TOTAL losses", round(inv_loss_share, 1), 55, 85, "%"))

    print("\n── Account / ATM reuse (mule network signal) ──────────")
    mule_rate = accounts.is_mule_account.mean() * 100
    results.append(check("Mule account pool size", round(mule_rate, 1), 14, 22, "%"))
    hotspot_rate = atms.is_known_hotspot.mean() * 100
    results.append(check("ATM hotspot pool size", round(hotspot_rate, 1), 14, 22, "%"))

    fraud_txns = transactions[transactions.complaint_id.notna()]
    mule_ids = set(accounts.loc[accounts.is_mule_account, "account_id"])
    reuse_in_data = fraud_txns.to_account_id.isin(mule_ids).mean() * 100
    results.append(check("Actual mule-account hit rate in fraud txns", round(reuse_in_data, 1), 60, 90, "%"))

    print("\n── Chain structure ─────────────────────────────────────")
    avg_hops = fraud_txns.groupby("complaint_id").hop_number.max().mean()
    results.append(check("Average hops per fraud chain", round(avg_hops, 2), 1.8, 3.2))

    print("\n── Noise / realism ─────────────────────────────────────")
    missing_district = complaints.victim_district.isna().mean() * 100
    results.append(check("Missing victim_district rate", round(missing_district, 1), 3, 10, "%"))
    bg_txn_rate = transactions.complaint_id.isna().mean() * 100
    results.append(check("Background (non-fraud) txn share", round(bg_txn_rate, 1), 8, 18, "%"))
    dupe_count = complaints.complaint_id.astype(str).str.contains("-DUP").sum()
    results.append(check("Duplicate complaints present", dupe_count, 1, n_complaints))

    print("\n── Ground truth integrity ──────────────────────────────")
    financial = complaints[complaints.amount_lost > 0]
    gt_coverage = financial.true_cashout_atm_id.notna().mean() * 100
    results.append(check("Financial complaints WITH ground truth", round(gt_coverage, 1), 95, 100, "%"))
    non_financial = complaints[complaints.amount_lost == 0]
    gt_leak = non_financial.true_cashout_atm_id.notna().mean() * 100
    results.append(check("Non-financial complaints WITHOUT ground truth (leak check)", round(gt_leak, 1), 0, 0, "%"))

    print(f"\n{'='*60}")
    passed = sum(results)
    print(f"RESULT: {passed}/{len(results)} checks passed")
    print(f"{'='*60}")

    return complaints, accounts, atms, transactions, withdrawals


if __name__ == "__main__":
    validate()