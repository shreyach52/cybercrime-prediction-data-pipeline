"""
STEP 2 — SCHEMA DESIGN

This file defines every entity as a Python `dataclass`. A dataclass
is just a plain class whose only job is to hold fields — Python
auto-generates the __init__ for you. This becomes your literal
INTERFACE CONTRACT: the field names and types here are what B, C, D
and E will code against, so get consensus on this file before you
generate a single row of data.

Key design decision (flagged in our earlier discussion): every
Complaint has a HIDDEN ground-truth block — the actual cash-out
location/time we simulated — which is stored separately from the
"visible" fields the models are allowed to learn from. If you leak
ground truth into the visible features, B/C's models will look
artificially perfect and E's scorecard becomes meaningless.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── ENTITY 1: Complaint ──────────────────────────────────────────────
# One row = one NCRP-style complaint filed by a victim.
@dataclass
class Complaint:
    complaint_id: str              # e.g. "CMP-000001"
    filed_at: datetime             # when the complaint was filed
    fraud_category: str            # NCRP major head, e.g. "Financial Fraud"
    fraud_subcategory: str         # NCRP sub-head, e.g. "UPI Related Frauds"
    amount_lost: float             # rupees
    victim_district: str           # where the VICTIM is located
    victim_state: str
    payment_mode: str              # "UPI", "Card", "Net Banking", "Wallet"
    reported_delay_hours: float    # time between fraud happening and being reported
                                    # (this is what makes "urgency" scoring meaningful —
                                    # a complaint reported 2 hrs late is far more urgent
                                    # to act on than one reported 5 days late)

    # linked entities (filled in once we generate accounts/transactions)
    victim_account_id: Optional[str] = None
    suspect_account_id: Optional[str] = None

    # --- HIDDEN GROUND TRUTH — never expose to B/C's models directly ---
    true_cashout_atm_id: Optional[str] = None
    true_cashout_window_start: Optional[datetime] = None
    true_cashout_window_end: Optional[datetime] = None


# ── ENTITY 2: Account ────────────────────────────────────────────────
# Bank accounts — both victim accounts (used once, clean) and mule
# accounts (reused across many complaints — this is what B's graph
# model is supposed to detect).
@dataclass
class Account:
    account_id: str
    account_holder_name: str
    bank_name: str
    account_type: str              # "Savings", "Current"
    opened_at: datetime
    is_mule_account: bool          # ground truth flag — hidden from B's model,
                                    # used only by E's scorecard to check if B
                                    # correctly flagged it as high-risk
    home_district: str
    device_fingerprint: str        # simulated device/IP hash — mule rings often
                                    # reuse the same device across "different"
                                    # accounts, giving B a second reuse signal
                                    # independent of transaction links


# ── ENTITY 3: ATM ────────────────────────────────────────────────────
# Physical cash-out points. Some ATMs get reused disproportionately
# by fraud rings (the 15-20% reuse mechanic) — this is what makes
# C's spatial hotspot model meaningful to build.
@dataclass
class ATM:
    atm_id: str
    bank_name: str
    district: str
    latitude: float
    longitude: float
    is_known_hotspot: bool         # ground truth — hidden, used for grading only


# ── ENTITY 4: Transaction ────────────────────────────────────────────
# A money movement between two accounts (victim → mule → mule → ...).
# Fraud chains are rarely one hop — money usually bounces through
# 2-4 mule accounts before cash-out, which is exactly the structure
# B's graph/network model needs to exist in the data to detect.
@dataclass
class Transaction:
    transaction_id: str
    complaint_id: str              # which complaint this transaction chain belongs to
    from_account_id: str
    to_account_id: str
    amount: float
    timestamp: datetime
    hop_number: int                # 1 = victim's money leaving, 2 = first mule hop, etc.


# ── ENTITY 5: Withdrawal ─────────────────────────────────────────────
# The final cash-out event — money leaving the banking system at an ATM.
# This is literally the event the whole project is trying to predict
# in advance.
@dataclass
class Withdrawal:
    withdrawal_id: str
    complaint_id: str
    account_id: str                # which mule account withdrew
    atm_id: str
    amount: float
    withdrawn_at: datetime
