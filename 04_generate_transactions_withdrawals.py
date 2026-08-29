"""
STEP 5 — GENERATE TRANSACTIONS + WITHDRAWALS

For every complaint that involves real money (amount_lost > 0), we
simulate the actual fraud chain:

    victim account (clean)
        --[hop 1]--> mule account
        --[hop 2]--> mule account  (0-3 more hops, using pick_account
        --[hop 3]--> mule account   from Step 4 so reuse is realistic)
        --[final]--> WITHDRAWAL at an ATM (using pick_atm from Step 4)

Timing logic:
  - fraud_time = filed_at - reported_delay_hours (the money actually
    left BEFORE the complaint was filed — that delay is why urgency
    scoring matters in the real system)
  - each hop happens fast after the previous one (mule networks move
    money quickly, usually within minutes-to-a-few-hours, to cash out
    before banks can freeze anything)
  - the withdrawal happens shortly after the last hop

The withdrawal's time and ATM become the HIDDEN GROUND TRUTH written
back onto the complaint (true_cashout_atm_id, true_cashout_window_*).
This is never exposed as a "feature" — it's the answer key Member E's
scorecard grades predictions against.
"""

import random
import numpy as np
import pandas as pd
from datetime import timedelta

# reuse the pickers we already built and validated in Step 4
from importlib import import_module
step4 = import_module("03_generate_accounts_atms")
pick_account = step4.pick_account
pick_atm = step4.pick_atm

random.seed(42)
np.random.seed(42)


def generate_transactions_and_withdrawals(complaints_df: pd.DataFrame,
                                           accounts_df: pd.DataFrame,
                                           atms_df: pd.DataFrame):
    clean_accounts = accounts_df.loc[~accounts_df.is_mule_account, "account_id"].tolist()

    transaction_rows = []
    withdrawal_rows = []
    txn_counter = 1
    wd_counter = 1

    # we'll build updated copies of these ground-truth fields, then
    # write them back onto the complaints dataframe at the end
    victim_acc_col, suspect_acc_col = [], []
    true_atm_col, true_start_col, true_end_col = [], [], []

    for _, complaint in complaints_df.iterrows():
        if complaint.amount_lost <= 0:
            # non-financial complaint (e.g. stalking) — no money chain at all
            victim_acc_col.append(None)
            suspect_acc_col.append(None)
            true_atm_col.append(None)
            true_start_col.append(None)
            true_end_col.append(None)
            continue

        fraud_time = complaint.filed_at - timedelta(hours=complaint.reported_delay_hours)

        victim_account = random.choice(clean_accounts)
        n_hops = random.choices([1, 2, 3, 4], weights=[15, 40, 30, 15], k=1)[0]

        current_account = victim_account
        current_time = fraud_time
        current_amount = complaint.amount_lost

        first_hop_account = None
        for hop in range(1, n_hops + 1):
            next_account = pick_account(accounts_df)  # weighted: ~75% pulls from mule pool
            # small cut taken at each hop (layering), amount shrinks slightly
            current_amount = round(current_amount * random.uniform(0.92, 0.99), 2)
            # each hop happens fast: 5 minutes to 6 hours after the previous one
            current_time = current_time + timedelta(minutes=random.randint(5, 360))

            transaction_rows.append({
                "transaction_id": f"TXN-{txn_counter:07d}",
                "complaint_id": complaint.complaint_id,
                "from_account_id": current_account,
                "to_account_id": next_account,
                "amount": current_amount,
                "timestamp": current_time,
                "hop_number": hop,
            })
            txn_counter += 1

            if hop == 1:
                first_hop_account = next_account
            current_account = next_account

        # final cash-out: the last account in the chain withdraws at an ATM
        withdrawal_atm = pick_atm(atms_df)  # weighted: ~75% pulls from hotspot pool
        withdrawal_time = current_time + timedelta(minutes=random.randint(5, 120))

        withdrawal_rows.append({
            "withdrawal_id": f"WD-{wd_counter:06d}",
            "complaint_id": complaint.complaint_id,
            "account_id": current_account,
            "atm_id": withdrawal_atm,
            "amount": current_amount,
            "withdrawn_at": withdrawal_time,
        })
        wd_counter += 1

        # ground truth window: +/- 30 min around the actual withdrawal time —
        # this is deliberately a WINDOW, not a point, because "predict the
        # exact minute" isn't realistic; a 60-min actionable window is
        victim_acc_col.append(victim_account)
        suspect_acc_col.append(first_hop_account)
        true_atm_col.append(withdrawal_atm)
        true_start_col.append(withdrawal_time - timedelta(minutes=30))
        true_end_col.append(withdrawal_time + timedelta(minutes=30))

    transactions_df = pd.DataFrame(transaction_rows)
    withdrawals_df = pd.DataFrame(withdrawal_rows)

    # write ground truth back onto a COPY of complaints (never mutate in place silently)
    updated_complaints = complaints_df.copy()
    updated_complaints["victim_account_id"] = victim_acc_col
    updated_complaints["suspect_account_id"] = suspect_acc_col
    updated_complaints["true_cashout_atm_id"] = true_atm_col
    updated_complaints["true_cashout_window_start"] = true_start_col
    updated_complaints["true_cashout_window_end"] = true_end_col

    return updated_complaints, transactions_df, withdrawals_df


if __name__ == "__main__":
    # pull in Steps 3 and 4 to get complaints/accounts/atms to link together
    step3 = import_module("02_generate_complaints")
    complaints = step3.generate_complaints(n=200)
    accounts = step4.generate_accounts(n=300)
    atms = step4.generate_atms(n=150)

    complaints, transactions, withdrawals = generate_transactions_and_withdrawals(
        complaints, accounts, atms
    )

    print(f"Complaints: {len(complaints)}")
    print(f"Complaints with a real cash chain: {complaints.true_cashout_atm_id.notna().sum()}")
    print(f"Transactions generated: {len(transactions)}")
    print(f"Withdrawals generated: {len(withdrawals)}")
    print()

    print("Average hops per fraud chain:", round(transactions.groupby("complaint_id").hop_number.max().mean(), 2))
    print()

    print("Top 5 most-reused mule accounts in transactions (proves reuse pattern):")
    print(transactions["to_account_id"].value_counts().head(5))
    print()

    print("Top 5 most-reused ATMs in withdrawals (proves hotspot pattern):")
    print(withdrawals["atm_id"].value_counts().head(5))
    print()

    print("Sample complaint with linked ground truth:")
    sample = complaints[complaints.true_cashout_atm_id.notna()].iloc[0]
    print(sample[["complaint_id", "filed_at", "victim_account_id", "suspect_account_id",
                   "true_cashout_atm_id", "true_cashout_window_start", "true_cashout_window_end"]])