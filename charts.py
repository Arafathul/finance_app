"""
charts.py — one function per chart, all returning a plotly Figure.

──────────────────────────────────────────────────────────────────────────────
HOW TO ADD A NEW CHART
──────────────────────────────────────────────────────────────────────────────
1. Prototype it in explore.ipynb with matplotlib or bokeh as usual.

2. Add a function here that accepts a DataFrame and returns a go.Figure:

       def build_my_chart(df: pd.DataFrame) -> go.Figure:
           fig = go.Figure()
           # ... your plotly traces ...
           fig.update_layout(title="My Chart", template="plotly_dark")
           return fig

3. Register it in app.py by adding one entry to CHARTS:

       {"title": "My Chart", "build": charts.build_my_chart},

That's it. Run `python app.py` and the chart appears at http://localhost:8050.
──────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import BUFFER_DAYS, HALVINGS, NEXT_HALVING, compute_emas, compute_rsi

CYCLE_COLORS = ["#f59e0b", "#10b981", "#3b82f6", "#ec4899"]


# ── Chart 1: Halving cycle overview ──────────────────────────────────────────
def build_halving_chart(df: pd.DataFrame) -> go.Figure:
    """Log-scale BTC price annotated with halving events, cycle windows, peaks and lows."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="BTC/USD",
            line=dict(color="rgba(255,255,255,0.45)", width=1),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        )
    )

    halving_keys = list(HALVINGS.keys())
    end_dates = [HALVINGS[k]["date"] for k in halving_keys[1:]] + [NEXT_HALVING]
    peak_legend_added = False
    low_legend_added = False

    for i, (cycle_name, info) in enumerate(HALVINGS.items()):
        h_date = info["date"]
        start = h_date + pd.Timedelta(days=BUFFER_DAYS)
        end = end_dates[i] - pd.Timedelta(days=BUFFER_DAYS)
        color = CYCLE_COLORS[i % len(CYCLE_COLORS)]

        fig.add_vrect(
            x0=start,
            x1=min(end, df.index[-1]),
            fillcolor=color,
            opacity=0.06,
            layer="below",
            line_width=0,
        )
        fig.add_vline(
            x=h_date.timestamp() * 1000,
            line=dict(color=color, width=1.5, dash="dash"),
            annotation_text=cycle_name,
            annotation_position="top right",
            annotation=dict(font_color=color, font_size=11),
        )

        window = df.loc[start : min(end, df.index[-1])]
        if window.empty:
            continue

        peak_idx = window["Close"].idxmax()
        low_idx = window["Close"].idxmin()
        peak_p = window.loc[peak_idx, "Close"]
        low_p = window.loc[low_idx, "Close"]

        fig.add_trace(
            go.Scatter(
                x=[peak_idx],
                y=[peak_p],
                mode="markers+text",
                marker=dict(color="#ef4444", size=10, symbol="triangle-up"),
                text=[f"${peak_p:,.0f}"],
                textposition="top center",
                textfont=dict(color="#ef4444", size=10),
                name="Cycle Peak",
                showlegend=not peak_legend_added,
                hovertemplate="%{x|%Y-%m-%d}<br>Peak: $%{y:,.0f}<extra></extra>",
            )
        )
        peak_legend_added = True

        fig.add_trace(
            go.Scatter(
                x=[low_idx],
                y=[low_p],
                mode="markers+text",
                marker=dict(color="#60a5fa", size=10, symbol="triangle-down"),
                text=[f"${low_p:,.0f}"],
                textposition="bottom center",
                textfont=dict(color="#60a5fa", size=10),
                name="Cycle Low",
                showlegend=not low_legend_added,
                hovertemplate="%{x|%Y-%m-%d}<br>Low: $%{y:,.0f}<extra></extra>",
            )
        )
        low_legend_added = True

    fig.update_layout(
        title="Bitcoin Price — Halving Cycles (Log Scale)",
        yaxis_type="log",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12),
        height=550,
        margin=dict(t=60, b=80),
    )
    return fig


# ── Chart 2: RSI strategy ─────────────────────────────────────────────────────
def build_rsi_chart(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """Price with EMA 50/200, RSI subplot, and buy/sell crossover signals."""
    df = compute_emas(df)
    df["RSI"] = compute_rsi(df["Close"])
    price_title = f"{ticker} — Price & EMAs" if ticker else "Price & EMAs"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
        subplot_titles=(price_title, "RSI (14)"),
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="BTC/USD",
            line=dict(color="#f59e0b", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["EMA_50"], mode="lines", name="EMA 50",
                   line=dict(color="#10b981", width=1, dash="dash")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["EMA_200"], mode="lines", name="EMA 200",
                   line=dict(color="#3b82f6", width=1, dash="dash")),
        row=1, col=1,
    )

    buy = df[(df["RSI"].shift(1) < 30) & (df["RSI"] >= 30)]
    sell = df[(df["RSI"].shift(1) > 70) & (df["RSI"] <= 70)]

    fig.add_trace(
        go.Scatter(x=buy.index, y=buy["Close"], mode="markers", name="Buy Signal",
                   marker=dict(color="#10b981", size=10, symbol="triangle-up"),
                   hovertemplate="%{x|%Y-%m-%d}<br>Buy @ $%{y:,.0f}<extra></extra>"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=sell.index, y=sell["Close"], mode="markers", name="Sell Signal",
                   marker=dict(color="#ef4444", size=10, symbol="triangle-down"),
                   hovertemplate="%{x|%Y-%m-%d}<br>Sell @ $%{y:,.0f}<extra></extra>"),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI",
                   line=dict(color="#a78bfa", width=1.5),
                   hovertemplate="%{x|%Y-%m-%d}<br>RSI: %{y:.1f}<extra></extra>"),
        row=2, col=1,
    )

    fig.add_hrect(y0=70, y1=100, fillcolor="#ef4444", opacity=0.05, layer="below", line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="#10b981", opacity=0.05, layer="below", line_width=0, row=2, col=1)
    fig.add_hline(y=70, line=dict(color="#ef4444", dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#10b981", dash="dash", width=1), row=2, col=1)

    fig.update_layout(
        title=f"RSI Strategy — {ticker if ticker else 'Price'}, EMAs & Signals",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12),
        height=600,
        margin=dict(t=60, b=80),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    return fig


# ── ADD YOUR NEW CHARTS BELOW ─────────────────────────────────────────────────
#
# def build_volume_chart(df: pd.DataFrame) -> go.Figure:
#     fig = go.Figure()
#     fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
#                          marker_color="#f59e0b"))
#     fig.update_layout(title="BTC Daily Volume", template="plotly_dark")
#     return fig
