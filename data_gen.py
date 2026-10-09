"""Synthetic B2B CRM data generator for the Revenue Operations Funnel & Forecast dashboard.

================================  SYNTHETIC DATA  ================================
Everything produced here is FAKE. No real company, customer, deal, or Salesforce
export is used. The generator is seeded (SEED = 42) so the numbers are identical
on every run. The shapes imitate a Salesforce-style opportunity/lead pipeline so
the analytics are realistic, but none of it is real data.
=================================================================================

Author: Leo Chan - github.com/GoldenCrate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42

# Canonical funnel stages, in order.
FUNNEL_STAGES = ["Lead", "MQL", "SQL", "Opportunity", "Closed-Won"]

# Open pipeline stages used for forecasting (everything not yet closed).
OPEN_STAGES = ["MQL", "SQL", "Opportunity"]

LEAD_SOURCES = [
    "Inbound - Organic",
    "Inbound - Paid Search",
    "Outbound - SDR",
    "Partner / Reseller",
    "Events / Conference",
    "Product-Led (Free Tier)",
]

SEGMENTS = ["SMB", "Mid-Market", "Enterprise"]
REGIONS = ["AMER", "EMEA", "APAC"]

# Probability that a deal that reaches a given stage advances to the next one.
# Tuned so each lead source has a different quality profile (used by the
# attribution view). Keys are (source) -> stage advance multipliers.
SOURCE_QUALITY = {
    "Inbound - Organic": 1.15,
    "Inbound - Paid Search": 0.95,
    "Outbound - SDR": 0.85,
    "Partner / Reseller": 1.25,
    "Events / Conference": 1.00,
    "Product-Led (Free Tier)": 1.10,
}

# Baseline per-stage advance probability (reaching the NEXT stage).
BASE_ADVANCE = {
    "Lead": 0.45,        # Lead -> MQL
    "MQL": 0.55,         # MQL -> SQL
    "SQL": 0.60,         # SQL -> Opportunity
    "Opportunity": 0.35, # Opportunity -> Closed-Won (win rate at last stage)
}

# Typical deal size by segment (mean, std) in USD.
SEGMENT_DEAL = {
    "SMB": (12_000, 4_000),
    "Mid-Market": (55_000, 18_000),
    "Enterprise": (180_000, 70_000),
}

# Default weighted-forecast probabilities by stage (RevOps-style).
STAGE_WIN_PROB = {
    "MQL": 0.10,
    "SQL": 0.25,
    "Opportunity": 0.55,
    "Closed-Won": 1.00,
}

# Quarterly booked-revenue quota the synthetic team is carrying.
DEFAULT_QUARTER_QUOTA = 2_500_000


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_opportunities(n: int = 1800, seed: int = SEED) -> pd.DataFrame:
    """Generate a synthetic opportunity/lead table.

    Each row is one opportunity that entered the funnel. We simulate how far it
    progressed, giving every row a ``max_stage`` (the furthest stage reached).
    """
    rng = _rng(seed)

    # Spread creation dates across ~12 months ending "today" so we can compute
    # sales-cycle length and have a current open pipeline.
    start = pd.Timestamp("2025-01-01")
    created_offsets = rng.integers(0, 330, size=n)
    created_date = start + pd.to_timedelta(created_offsets, unit="D")

    segment = rng.choice(SEGMENTS, size=n, p=[0.55, 0.30, 0.15])
    source = rng.choice(LEAD_SOURCES, size=n)
    region = rng.choice(REGIONS, size=n, p=[0.5, 0.3, 0.2])

    owners = [f"AE-{i:02d}" for i in range(1, 13)]
    owner = rng.choice(owners, size=n)

    # Simulate furthest stage reached for each opportunity.
    max_stage_idx = np.zeros(n, dtype=int)  # index into FUNNEL_STAGES
    for i in range(n):
        stage_i = 0  # starts as "Lead"
        quality = SOURCE_QUALITY[source[i]]
        for s in ["Lead", "MQL", "SQL", "Opportunity"]:
            p = min(0.95, BASE_ADVANCE[s] * quality)
            if rng.random() < p:
                stage_i += 1
            else:
                break
        max_stage_idx[i] = stage_i

    max_stage = np.array(FUNNEL_STAGES)[max_stage_idx]

    # Deal amount by segment.
    amount = np.zeros(n)
    for seg, (mu, sd) in SEGMENT_DEAL.items():
        mask = segment == seg
        amount[mask] = np.clip(rng.normal(mu, sd, size=mask.sum()), 2_000, None)
    amount = np.round(amount, -2)  # round to nearest 100

    # Current stage = max stage reached (open deals sit at their furthest stage).
    stage = max_stage.copy()

    is_won = stage == "Closed-Won"
    is_closed = is_won  # in this synthetic model, lost deals stop at max stage

    # Sales cycle (days) only meaningful for closed-won deals.
    cycle_mean = {"SMB": 35, "Mid-Market": 70, "Enterprise": 120}
    sales_cycle_days = np.full(n, np.nan)
    close_date = pd.Series([pd.NaT] * n)
    for i in range(n):
        if is_won[i]:
            base = cycle_mean[segment[i]]
            days = int(max(7, rng.normal(base, base * 0.3)))
            sales_cycle_days[i] = days
            close_date.iloc[i] = created_date[i] + pd.Timedelta(days=days)

    # Expected close date for OPEN deals (for forecasting) — projected forward.
    expected_close = pd.Series([pd.NaT] * n)
    for i in range(n):
        if not is_closed[i] and stage[i] != "Lead":
            base = cycle_mean[segment[i]]
            proj = int(max(5, rng.normal(base * 0.6, base * 0.2)))
            expected_close.iloc[i] = created_date[i] + pd.Timedelta(days=proj)

    df = pd.DataFrame(
        {
            "opp_id": [f"OPP-{i:05d}" for i in range(1, n + 1)],
            "created_date": created_date,
            "close_date": close_date.values,
            "expected_close_date": expected_close.values,
            "stage": stage,
            "amount": amount,
            "lead_source": source,
            "segment": segment,
            "region": region,
            "owner": owner,
            "is_won": is_won,
            "is_closed": is_closed,
            "sales_cycle_days": sales_cycle_days,
        }
    )
    return df


def ae_quota_table(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Synthetic per-AE quarterly quota + attainment table."""
    rng = _rng(seed + 1)
    owners = sorted(df["owner"].unique())
    quota = rng.integers(180_000, 260_000, size=len(owners))
    won = (
        df[df["is_won"]].groupby("owner")["amount"].sum().reindex(owners).fillna(0.0)
    )
    out = pd.DataFrame(
        {
            "owner": owners,
            "quota": quota,
            "booked": won.values,
        }
    )
    out["attainment"] = out["booked"] / out["quota"]
    return out


if __name__ == "__main__":
    opps = generate_opportunities()
    print("SYNTHETIC DATA — not real. Seed =", SEED)
    print(opps.shape)
    print(opps.head())
    print("\nStage distribution:")
    print(opps["stage"].value_counts())
    print("\nAE quota table:")
    print(ae_quota_table(opps).head())
