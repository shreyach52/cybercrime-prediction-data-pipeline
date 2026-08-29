"""
PRIMER — the 4 techniques you'll see everywhere in this project.
Run this file top to bottom and read the comments. Nothing here
is final code — it's just to build intuition before we write the
real generator.
"""

import random
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker("en_IN")  # "en_IN" = Indian-flavored names/addresses/phone numbers

# ── TECHNIQUE 1: weighted random choice ──────────────────────────────
# Real fraud isn't 50/50 split across categories — e.g. UPI fraud is
# far more common than, say, crypto fraud. We fake that by giving
# each category a WEIGHT (a relative probability), not equal odds.

fraud_types = ["UPI Fraud", "Credit/Debit Card Fraud", "Internet Banking Fraud", "Crypto Fraud"]
weights =     [        50 ,                       25 ,                      20 ,             5 ]
# ^ these numbers don't need to sum to 100 — random.choices() normalizes them.
# Roughly: 50% UPI, 25% card, 20% netbanking, 5% crypto — this will get replaced
# with real NCRP-published stats in the research step, this is just a placeholder.

sample_fraud_type = random.choices(fraud_types, weights=weights, k=1)[0]
print("One random fraud type (weighted):", sample_fraud_type)

# ── TECHNIQUE 2: sampling from a statistical distribution ───────────
# Fraud amounts aren't uniform random either — most cybercrime losses
# are small-to-medium, with a long tail of a few huge ones. A LOGNORMAL
# distribution naturally produces that shape (many small values, few
# huge ones), which is why it's the standard choice for modeling money.

amounts = np.random.lognormal(mean=8.5, sigma=1.2, size=5)  # size=5 → generate 5 samples
print("\n5 sample fraud amounts (lognormal, rupees):", [round(a, 2) for a in amounts])

# ── TECHNIQUE 3: Faker for realistic-looking fake identity data ─────
print("\nFaker examples:")
print(" fake name:     ", fake.name())
print(" fake phone:     ", fake.phone_number())
print(" fake date/time: ", fake.date_time_between(start_date="-30d", end_date="now"))

# ── TECHNIQUE 4: building a table (DataFrame) from a list of dicts ──
# Every "generate N records" function we write will follow this exact
# pattern: build a Python list of dicts (one dict = one row), then hand
# it to pandas to turn into a table you can inspect, filter, and export.

rows = []
for i in range(5):
    rows.append({
        "id": i,
        "name": fake.name(),
        "fraud_type": random.choices(fraud_types, weights=weights, k=1)[0],
        "amount": round(np.random.lognormal(mean=8.5, sigma=1.2), 2),
    })

df = pd.DataFrame(rows)
print("\nA tiny 5-row synthetic table:")
print(df)
