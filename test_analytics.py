"""Unit tests for analytics.py (pytest).

Run: pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import analytics as a
from data_gen import generate_opportunities


@pytest.fixture(scope="module")
def df():
    return generate_opportunities(n=1800, seed=42)


# ------------------------------------------------------------------- determinism
def test_data_is_deterministic():
    d1 = generate_opportunities(n=500, seed=42)
    d2 = generate_opportunities(n=500, seed=42)
    pd.testing.assert_frame_equal(d1, d2)


def test_expected_columns(df):
    for col in ["opp_id", "stage", "amount", "lead_source", "is_won", "is_closed"]:
        assert col in df.columns


# -------------------------------------------------------------------------- KPIs
def test_kpi_summary_ranges(df):
    k = a.kpi_summary(df)
    assert 0.0 <= k["win_rate"] <= 1.0
    assert k["open_pipeline"] >= 0
    assert k["avg_deal_size"] > 0
    assert k["median_sales_cycle_days"] > 0
    assert k["n_won"] >= 0


def test_quota_attainment():
    assert a.quota_attainment(500_000, 1_000_000) == 0.5
    assert a.quota_attainment(100, 0) == 0.0


# ------------------------------------------------------------------------ Funnel
def test_funnel_counts_monotonic(df):
    fc = a.funnel_counts(df)
    counts = fc["count"].tolist()
    # Cumulative funnel: each stage has <= the previous stage's count.
    assert counts == sorted(counts, reverse=True)
    assert counts[0] == len(df)  # everyone reached "Lead"


def test_funnel_conversion_bounds(df):
    conv = a.funnel_conversion(df)
    assert (conv["step_conversion"] >= 0).all()
    assert (conv["step_conversion"] <= 1.0 + 1e-9).all()
    assert conv.iloc[0]["step_conversion"] == 1.0
    # Overall conversion is non-increasing down the funnel.
    oc = conv["overall_conversion"].tolist()
    assert oc == sorted(oc, reverse=True)


def test_stage_velocity_sums_to_cycle(df):
    sv = a.stage_velocity(df)
    assert not sv.empty
    won = df[df["is_won"]]
    total = won["sales_cycle_days"].mean()
    assert sv["avg_days"].sum() == pytest.approx(total, rel=0.02)


# ----------------------------------------------------------- Pipeline & forecast
def test_weighted_forecast(df):
    wf = a.weighted_forecast(df)
    assert set(wf["stage"]) == set(a.OPEN_STAGES)
    for _, row in wf.iterrows():
        assert row["weighted_forecast"] == pytest.approx(
            row["open_pipeline"] * row["win_prob"], rel=1e-6
        )
        assert row["weighted_forecast"] <= row["open_pipeline"] + 1e-6


def test_pipeline_coverage():
    df = pd.DataFrame(
        {
            "stage": ["SQL", "Opportunity"],
            "amount": [300_000, 300_000],
            "is_closed": [False, False],
        }
    )
    # pipeline = 600k, gap = 200k -> coverage 3.0x
    assert a.pipeline_coverage(df, 200_000) == pytest.approx(3.0)
    assert a.pipeline_coverage(df, 0) == 0.0


def test_forecast_vs_quota(df):
    fv = a.forecast_vs_quota(df, quota=2_500_000)
    assert fv["total_forecast"] == pytest.approx(
        fv["booked"] + fv["weighted_open_forecast"]
    )
    assert fv["quota_gap"] >= 0
    assert fv["attainment_forecast"] >= 0


# ----------------------------------------------------------------- Attribution
def test_channel_performance(df):
    cp = a.channel_performance(df, by="lead_source")
    assert "win_rate" in cp.columns
    assert (cp["win_rate"] >= 0).all() and (cp["win_rate"] <= 1.0).all()
    assert cp["total_opps"].sum() == len(df)
    # Sorted by booked revenue descending.
    assert cp["booked_revenue"].tolist() == sorted(
        cp["booked_revenue"].tolist(), reverse=True
    )


def test_channel_performance_bad_column(df):
    with pytest.raises(ValueError):
        a.channel_performance(df, by="not_a_column")


def test_fmt_money():
    assert a.fmt_money(1_200_000) == "$1.2M"
    assert a.fmt_money(45_000) == "$45K"
    assert a.fmt_money(500) == "$500"
