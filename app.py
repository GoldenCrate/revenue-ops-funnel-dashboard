"""B2B Revenue Operations Funnel & Forecast — Streamlit dashboard.

A RevOps analyst's working view over a Salesforce-style pipeline: KPIs, the
Lead -> Closed-Won funnel, pipeline health & a weighted forecast vs. quota, and
channel/segment attribution. Every view ends with a plain-English
"So what / recommendation".

ALL DATA IS SYNTHETIC (seeded, generated in data_gen.py). Nothing here is real.

Author: Leo Chan - github.com/GoldenCrate
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as a
from data_gen import DEFAULT_QUARTER_QUOTA, ae_quota_table, generate_opportunities

# --------------------------------------------------------------------- styling
# Storytelling-with-Data (Knaflic): muted greys for context, ONE accent color
# to carry attention, direct labels over legends, no chartjunk.
GREY = "#B3B3B3"
GREY_DARK = "#636363"
ACCENT = "#1F77B4"      # the one highlight color
POSITIVE = "#2CA02C"
WARNING = "#D62728"
TEXT = "#2B2B2B"

st.set_page_config(
    page_title="B2B RevOps Funnel & Forecast",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data():
    opps = generate_opportunities()
    return opps


def so_what(text: str) -> None:
    """Render a highlighted 'So what / recommendation' callout."""
    st.markdown(
        f"""
        <div style="background:#F2F7FB;border-left:4px solid {ACCENT};
        padding:0.75rem 1rem;border-radius:4px;margin-top:0.5rem;">
        <span style="font-weight:700;color:{ACCENT};">So what →</span>
        <span style="color:{TEXT};">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def money(x: float) -> str:
    return a.fmt_money(x)


# ----------------------------------------------------------------------- sidebar
df_all = load_data()

st.sidebar.title("RevOps controls")
st.sidebar.caption("All data is synthetic — seeded demo, not a real CRM.")

segments = st.sidebar.multiselect(
    "Segment", sorted(df_all["segment"].unique()), default=sorted(df_all["segment"].unique())
)
regions = st.sidebar.multiselect(
    "Region", sorted(df_all["region"].unique()), default=sorted(df_all["region"].unique())
)
sources = st.sidebar.multiselect(
    "Lead source",
    sorted(df_all["lead_source"].unique()),
    default=sorted(df_all["lead_source"].unique()),
)
quota = st.sidebar.number_input(
    "Quarterly quota ($)", value=int(DEFAULT_QUARTER_QUOTA), step=100_000
)

df = df_all[
    df_all["segment"].isin(segments)
    & df_all["region"].isin(regions)
    & df_all["lead_source"].isin(sources)
].copy()

if df.empty:
    st.warning("No opportunities match the current filters. Widen the selection.")
    st.stop()

st.title("B2B Revenue Operations — Funnel & Forecast")
st.caption(
    "Synthetic Salesforce-style pipeline • Lead → MQL → SQL → Opportunity → "
    "Closed-Won • built by Leo Chan (github.com/GoldenCrate)"
)

view = st.radio(
    "View",
    ["KPI Overview", "Sales Funnel", "Pipeline Health & Forecast", "Channel / Attribution"],
    horizontal=True,
    label_visibility="collapsed",
)


# =========================================================== 1. KPI OVERVIEW
if view == "KPI Overview":
    k = a.kpi_summary(df)
    attainment = a.quota_attainment(k["booked_revenue"], quota)

    st.subheader("Headline KPIs")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Open pipeline", money(k["open_pipeline"]), f'{k["n_open_opps"]} open deals')
    c2.metric("Win rate", f'{k["win_rate"] * 100:.0f}%')
    c3.metric("Avg deal size", money(k["avg_deal_size"]))
    c4.metric("Median sales cycle", f'{k["median_sales_cycle_days"]:.0f} days')
    c5.metric("Quota attainment", f"{attainment * 100:.0f}%", f'{money(k["booked_revenue"])} booked')

    st.divider()

    # Booked revenue by month (context in grey, latest-trend story).
    won = df[df["is_won"]].copy()
    won["month"] = pd.to_datetime(won["close_date"]).dt.to_period("M").dt.to_timestamp()
    by_month = won.groupby("month")["amount"].sum().reset_index()

    fig = go.Figure()
    fig.add_bar(
        x=by_month["month"], y=by_month["amount"],
        marker_color=ACCENT, name="Booked",
        hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra></extra>",
    )
    fig.update_layout(
        title="Booked (Closed-Won) revenue by month",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(showgrid=True, gridcolor="#EEE", tickprefix="$", title=None),
        xaxis=dict(title=None),
        margin=dict(t=50, b=20, l=20, r=20), height=360, showlegend=False,
        font=dict(color=TEXT),
    )
    st.plotly_chart(fig, use_container_width=True)

    gap = max(0, quota - k["booked_revenue"])
    so_what(
        f"Booked revenue is {money(k['booked_revenue'])} against a {money(quota)} "
        f"quota ({attainment*100:.0f}% attainment), leaving a {money(gap)} gap. "
        f"With a {k['win_rate']*100:.0f}% win rate and {money(k['avg_deal_size'])} "
        f"average deal, roughly {int(gap / max(k['avg_deal_size'],1)) + 1} more "
        "won deals close the quarter — focus reps on the highest-converting sources."
    )


# =========================================================== 2. SALES FUNNEL
elif view == "Sales Funnel":
    conv = a.funnel_conversion(df)
    st.subheader("Lead → Closed-Won funnel")

    # Horizontal funnel bars; weakest step highlighted in red (the story).
    steps = conv.iloc[1:].copy()  # drop the Lead->Lead 100%
    weakest_idx = steps["step_conversion"].idxmin()
    bar_colors = [
        ACCENT if s == "Lead" else (WARNING if i == weakest_idx else GREY)
        for i, s in zip(conv.index, conv["stage"])
    ]

    fig = go.Figure()
    fig.add_bar(
        y=conv["stage"], x=conv["count"], orientation="h",
        marker_color=bar_colors,
        text=[f"{c:,}" for c in conv["count"]],
        textposition="outside",
        hovertemplate="%{y}: %{x:,} deals<extra></extra>",
    )
    fig.update_layout(
        title="Deals reaching each stage (cumulative)",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(a.FUNNEL_STAGES))),
        xaxis=dict(showgrid=False, title=None),
        margin=dict(t=50, b=20, l=20, r=60), height=380, showlegend=False,
        font=dict(color=TEXT),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stage-to-stage conversion**")
        disp = conv.copy()
        disp["step_conversion"] = (disp["step_conversion"] * 100).round(0).astype(int).astype(str) + "%"
        disp["overall_conversion"] = (disp["overall_conversion"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            disp.rename(columns={
                "stage": "Stage", "count": "Deals",
                "step_conversion": "Step conv.", "overall_conversion": "Overall conv.",
            }),
            hide_index=True, use_container_width=True,
        )
    with c2:
        st.markdown("**Velocity — avg days per transition (won deals)**")
        sv = a.stage_velocity(df)
        st.dataframe(
            sv.rename(columns={"stage_transition": "Transition", "avg_days": "Avg days"}),
            hide_index=True, use_container_width=True,
        )

    worst = steps.loc[weakest_idx]
    prev_stage = a.FUNNEL_STAGES[a.FUNNEL_STAGES.index(worst["stage"]) - 1]
    so_what(
        f"The biggest drop-off is {prev_stage} → {worst['stage']} at only "
        f"{worst['step_conversion']*100:.0f}% conversion — the tightest point in the "
        "funnel. Fixing this stage (better qualification criteria or sales enablement) "
        "lifts every downstream stage, so it is the highest-leverage place to invest."
    )


# ================================================= 3. PIPELINE HEALTH & FORECAST
elif view == "Pipeline Health & Forecast":
    st.subheader("Pipeline health & weighted forecast")
    fv = a.forecast_vs_quota(df, quota)

    c1, c2, c3 = st.columns(3)
    c1.metric("Coverage ratio", f'{fv["coverage_ratio"]:.1f}x',
              "healthy ≈ 3x–4x")
    c2.metric("Weighted open forecast", money(fv["weighted_open_forecast"]))
    c3.metric("Forecast attainment", f'{fv["attainment_forecast"]*100:.0f}%',
              f'{money(fv["total_forecast"])} total')

    # Waterfall: Booked -> + weighted open -> Total vs quota line.
    wf = a.weighted_forecast(df)
    fig = go.Figure()
    fig.add_bar(
        x=["Booked"] + list(wf["stage"]) + ["Total forecast"],
        y=[fv["booked"]] + list(wf["weighted_forecast"]) + [fv["total_forecast"]],
        marker_color=[POSITIVE] + [GREY] * len(wf) + [ACCENT],
        text=[money(v) for v in [fv["booked"], *wf["weighted_forecast"], fv["total_forecast"]]],
        textposition="outside",
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
    )
    fig.add_hline(
        y=quota, line_dash="dash", line_color=WARNING,
        annotation_text=f"Quota {money(quota)}", annotation_position="top right",
        annotation_font_color=WARNING,
    )
    fig.update_layout(
        title="Weighted forecast build vs. quota",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(showgrid=True, gridcolor="#EEE", tickprefix="$", title=None),
        xaxis=dict(title=None),
        margin=dict(t=50, b=20, l=20, r=20), height=400, showlegend=False,
        font=dict(color=TEXT),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Weighted pipeline by stage**")
    disp = wf.copy()
    disp["open_pipeline"] = disp["open_pipeline"].map(money)
    disp["win_prob"] = (disp["win_prob"] * 100).astype(int).astype(str) + "%"
    disp["weighted_forecast"] = disp["weighted_forecast"].map(money)
    st.dataframe(
        disp.rename(columns={
            "stage": "Stage", "open_pipeline": "Open $",
            "win_prob": "Win prob.", "weighted_forecast": "Weighted $", "n_deals": "Deals",
        }),
        hide_index=True, use_container_width=True,
    )

    cov = fv["coverage_ratio"]
    verdict = ("healthy" if cov >= 3 else "thin — below the 3x rule of thumb")
    so_what(
        f"Weighted forecast is {money(fv['total_forecast'])} vs a {money(quota)} quota "
        f"({fv['attainment_forecast']*100:.0f}%). Coverage on the {money(fv['quota_gap'])} "
        f"remaining gap is {cov:.1f}x, which is {verdict}. "
        + ("Pipeline is sufficient — prioritise late-stage Opportunity deals to convert the forecast."
           if cov >= 3 else
           "Generate more top-of-funnel pipeline now; the current gap will not close on existing deals alone.")
    )


# ================================================= 4. CHANNEL / ATTRIBUTION
elif view == "Channel / Attribution":
    st.subheader("Channel & segment attribution")
    dim = st.selectbox(
        "Break down by", ["lead_source", "segment", "region"],
        format_func=lambda s: {"lead_source": "Lead source", "segment": "Segment", "region": "Region"}[s],
    )
    cp = a.channel_performance(df, by=dim)

    # Highlight the top revenue channel in accent, rest grey.
    top_label = cp.iloc[0][dim]
    colors = [ACCENT if v == top_label else GREY for v in cp[dim]]

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(
            x=cp["booked_revenue"], y=cp[dim], orientation="h",
            marker_color=colors,
            text=[money(v) for v in cp["booked_revenue"]], textposition="outside",
            hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
        )
        fig.update_layout(
            title="Booked revenue by channel",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False, tickprefix="$", title=None),
            yaxis=dict(autorange="reversed", title=None),
            margin=dict(t=50, b=20, l=20, r=60), height=400, showlegend=False,
            font=dict(color=TEXT),
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = go.Figure()
        fig2.add_bar(
            x=cp["win_rate"] * 100, y=cp[dim], orientation="h",
            marker_color=colors,
            text=[f"{v*100:.0f}%" for v in cp["win_rate"]], textposition="outside",
            hovertemplate="%{y}: %{x:.0f}% win rate<extra></extra>",
        )
        fig2.update_layout(
            title="Win rate by channel",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False, ticksuffix="%", title=None, range=[0, max(cp["win_rate"]*100)*1.3]),
            yaxis=dict(autorange="reversed", title=None),
            margin=dict(t=50, b=20, l=20, r=60), height=400, showlegend=False,
            font=dict(color=TEXT),
        )
        st.plotly_chart(fig2, use_container_width=True)

    disp = cp.copy()
    disp["win_rate"] = (disp["win_rate"] * 100).round(0).astype(int).astype(str) + "%"
    disp["booked_revenue"] = disp["booked_revenue"].map(money)
    st.dataframe(
        disp.rename(columns={
            dim: dim.replace("_", " ").title(), "total_opps": "Total opps",
            "closed": "Closed", "won": "Won", "booked_revenue": "Booked $", "win_rate": "Win rate",
        }),
        hide_index=True, use_container_width=True,
    )

    best_wr = cp.loc[cp["win_rate"].idxmax()]
    so_what(
        f"{top_label} drives the most booked revenue ({money(cp.iloc[0]['booked_revenue'])}), "
        f"while {best_wr[dim]} converts at the highest win rate "
        f"({best_wr['win_rate']*100:.0f}%). Shift budget toward the channels that are "
        "both high-volume and high-conversion, and re-qualify traffic from low-win-rate "
        "sources before it clogs the funnel."
    )

st.divider()
st.caption(
    "⚠️ Synthetic data only — generated with a fixed seed for demonstration. "
    "No real customer, deal, or CRM data is used. © Leo Chan — github.com/GoldenCrate"
)
