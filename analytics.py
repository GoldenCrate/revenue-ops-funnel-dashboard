"""Framework-free RevOps analytics.

Pure pandas/numpy functions — no Streamlit, no plotting. Everything here is
unit-testable (see test_analytics.py) and reused by app.py for display.

Author: Leo Chan - github.com/GoldenCrate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FUNNEL_STAGES = ["Lead", "MQL", "SQL", "Opportunity", "Closed-Won"]
OPEN_STAGES = ["MQL", "SQL", "Opportunity"]
STAGE_WIN_PROB = {"MQL": 0.10, "SQL": 0.25, "Opportunity": 0.55, "Closed-Won": 1.00}


# ----------------------------------------------------------------------------- KPIs
def kpi_summary(df: pd.DataFrame) -> dict:
    """Headline revenue-operations KPIs.

    Returns pipeline $ (open, un-weighted), win rate, average won deal size,
    median sales cycle (days), and total booked (Closed-Won) revenue.
    """
    closed = df[df["is_closed"]]
    won = df[df["is_won"]]
    open_deals = df[~df["is_closed"] & (df["stage"] != "Lead")]

    n_closed = len(closed)
    win_rate = (len(won) / n_closed) if n_closed else 0.0

    return {
        "open_pipeline": float(open_deals["amount"].sum()),
        "booked_revenue": float(won["amount"].sum()),
        "win_rate": float(win_rate),
        "avg_deal_size": float(won["amount"].mean()) if len(won) else 0.0,
        "median_sales_cycle_days": float(won["sales_cycle_days"].median())
        if len(won)
        else 0.0,
        "n_open_opps": int(len(open_deals)),
        "n_won": int(len(won)),
    }


def quota_attainment(booked_revenue: float, quota: float) -> float:
    """Booked revenue as a fraction of quota (0.0+ ; 1.0 == 100%)."""
    if quota <= 0:
        return 0.0
    return float(booked_revenue / quota)


# -------------------------------------------------------------------------- Funnel
def funnel_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count of opportunities that REACHED each stage (cumulative funnel).

    A deal whose furthest stage is 'SQL' is counted in Lead, MQL, and SQL.
    """
    reached_idx = df["stage"].map(lambda s: FUNNEL_STAGES.index(s))
    rows = []
    for i, stage in enumerate(FUNNEL_STAGES):
        count = int((reached_idx >= i).sum())
        rows.append({"stage": stage, "count": count})
    out = pd.DataFrame(rows)
    return out


def funnel_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Stage-to-stage conversion plus overall (top-of-funnel) conversion."""
    counts = funnel_counts(df).set_index("stage")["count"]
    top = counts.iloc[0]
    rows = []
    for i, stage in enumerate(FUNNEL_STAGES):
        c = int(counts.iloc[i])
        if i == 0:
            step = 1.0
        else:
            prev = counts.iloc[i - 1]
            step = (c / prev) if prev else 0.0
        overall = (c / top) if top else 0.0
        rows.append(
            {
                "stage": stage,
                "count": c,
                "step_conversion": float(step),
                "overall_conversion": float(overall),
            }
        )
    return pd.DataFrame(rows)


def stage_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Average days a won deal spends reaching each stage, by segment proxy.

    We only have end-to-end cycle for won deals, so we allocate the total cycle
    across stages using fixed weights. This gives a 'velocity' view of where
    deals spend time — a classic RevOps bottleneck diagnostic.
    """
    won = df[df["is_won"]]
    if won.empty:
        return pd.DataFrame(columns=["stage_transition", "avg_days"])

    avg_cycle = float(won["sales_cycle_days"].mean())
    # Fixed share of total cycle per transition (sums to 1.0).
    weights = {
        "Lead -> MQL": 0.15,
        "MQL -> SQL": 0.20,
        "SQL -> Opportunity": 0.30,
        "Opportunity -> Closed-Won": 0.35,
    }
    rows = [
        {"stage_transition": k, "avg_days": round(avg_cycle * w, 1)}
        for k, w in weights.items()
    ]
    return pd.DataFrame(rows)


# ------------------------------------------------------------- Pipeline & forecast
def weighted_forecast(
    df: pd.DataFrame, stage_prob: dict | None = None
) -> pd.DataFrame:
    """Probability-weighted pipeline forecast by stage.

    For each OPEN stage, multiply open pipeline $ by the stage win probability.
    """
    probs = stage_prob or STAGE_WIN_PROB
    open_deals = df[~df["is_closed"] & df["stage"].isin(OPEN_STAGES)]
    rows = []
    for stage in OPEN_STAGES:
        sub = open_deals[open_deals["stage"] == stage]
        pipeline = float(sub["amount"].sum())
        p = probs.get(stage, 0.0)
        rows.append(
            {
                "stage": stage,
                "open_pipeline": pipeline,
                "win_prob": p,
                "weighted_forecast": round(pipeline * p, 2),
                "n_deals": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def pipeline_coverage(df: pd.DataFrame, quota_gap: float) -> float:
    """Open pipeline $ divided by remaining quota gap (coverage ratio).

    RevOps rule of thumb: ~3x-4x coverage is healthy. Returns 0 if no gap.
    """
    open_deals = df[~df["is_closed"] & (df["stage"] != "Lead")]
    pipeline = float(open_deals["amount"].sum())
    if quota_gap <= 0:
        return 0.0
    return float(pipeline / quota_gap)


def forecast_vs_quota(df: pd.DataFrame, quota: float, stage_prob=None) -> dict:
    """Combine booked + weighted open forecast against quota."""
    booked = float(df[df["is_won"]]["amount"].sum())
    wf = weighted_forecast(df, stage_prob)
    weighted_open = float(wf["weighted_forecast"].sum())
    total_forecast = booked + weighted_open
    gap = max(0.0, quota - booked)
    return {
        "booked": booked,
        "weighted_open_forecast": weighted_open,
        "total_forecast": total_forecast,
        "quota": float(quota),
        "attainment_forecast": (total_forecast / quota) if quota else 0.0,
        "quota_gap": gap,
        "coverage_ratio": pipeline_coverage(df, gap),
    }


# ----------------------------------------------------------------- Attribution
def channel_performance(df: pd.DataFrame, by: str = "lead_source") -> pd.DataFrame:
    """Win rate, booked revenue, and deal counts by channel/segment.

    ``by`` can be 'lead_source', 'segment', or 'region'.
    """
    if by not in df.columns:
        raise ValueError(f"Unknown grouping column: {by}")

    g = df.groupby(by)
    closed = df[df["is_closed"]].groupby(by).size()
    won = df[df["is_won"]].groupby(by).size()
    booked = df[df["is_won"]].groupby(by)["amount"].sum()

    out = pd.DataFrame({"total_opps": g.size()})
    out["closed"] = closed.reindex(out.index).fillna(0).astype(int)
    out["won"] = won.reindex(out.index).fillna(0).astype(int)
    out["booked_revenue"] = booked.reindex(out.index).fillna(0.0)
    out["win_rate"] = np.where(out["closed"] > 0, out["won"] / out["closed"], 0.0)
    out = out.reset_index().sort_values("booked_revenue", ascending=False)
    return out.reset_index(drop=True)


def fmt_money(x: float) -> str:
    """Compact money formatter for display (e.g. $1.2M, $45.0K)."""
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"${x / 1_000:.0f}K"
    return f"${x:,.0f}"
