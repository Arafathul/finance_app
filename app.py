import time

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from dash import dash_table, dcc, html
from plotly.subplots import make_subplots

# ── App init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="BTC Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # expose Flask server for Gunicorn / Render

# ── Constants ─────────────────────────────────────────────────────────────────
HALVINGS = {
    "H1 (2012)": {"date": pd.Timestamp("2012-11-28"), "price": 12.35},
    "H2 (2016)": {"date": pd.Timestamp("2016-07-09"), "price": 650.63},
    "H3 (2020)": {"date": pd.Timestamp("2020-05-11"), "price": 8_821.42},
    "H4 (2024)": {"date": pd.Timestamp("2024-04-20"), "price": 63_850.00},
}
NEXT_HALVING = pd.Timestamp("2028-04-17")
BUFFER_DAYS = 60
CACHE_TTL = 3_600  # seconds

# ── Data layer ────────────────────────────────────────────────────────────────
_cache: dict = {}
_cache_times: dict = {}


def load_btc_data() -> pd.DataFrame:
    """Fetch BTC-USD daily history from yfinance with a 1-hour in-process cache."""
    key = "btc"
    now = time.time()
    if key in _cache and now - _cache_times.get(key, 0) < CACHE_TTL:
        return _cache[key]
    df = yf.Ticker("BTC-USD").history(period="max", interval="1d")
    df.index = df.index.tz_localize(None)
    _cache[key] = df
    _cache_times[key] = now
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_emas(df: pd.DataFrame, short: int = 50, long: int = 200) -> pd.DataFrame:
    df = df.copy()
    df["EMA_50"] = df["Close"].ewm(span=short, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=long, adjust=False).mean()
    return df


# ── Analytics ─────────────────────────────────────────────────────────────────
def get_cycle_analysis(df: pd.DataFrame) -> list[dict]:
    rows = []
    halving_keys = list(HALVINGS.keys())
    end_dates = [HALVINGS[k]["date"] for k in halving_keys[1:]] + [NEXT_HALVING]

    for i, (cycle_name, info) in enumerate(HALVINGS.items()):
        h_date = info["date"]
        start = h_date + pd.Timedelta(days=BUFFER_DAYS)
        end = end_dates[i] - pd.Timedelta(days=BUFFER_DAYS)
        window = df.loc[start:end]
        if window.empty:
            continue

        peak_idx = window["Close"].idxmax()
        low_idx = window["Close"].idxmin()
        peak = window.loc[peak_idx, "Close"]
        low = window.loc[low_idx, "Close"]

        rows.append(
            {
                "Cycle": cycle_name,
                "Halving Date": h_date.strftime("%Y-%m-%d"),
                "Price at Halving": f"${info['price']:,.2f}",
                "Peak Date": peak_idx.strftime("%Y-%m-%d"),
                "Days to Peak": (peak_idx - h_date).days,
                "Peak Price": f"${peak:,.0f}",
                "Multiplier": f"{peak / info['price']:.2f}x",
                "Low Date": low_idx.strftime("%Y-%m-%d"),
                "Low Price": f"${low:,.0f}",
                "Drawdown": f"{((low - peak) / peak * 100):.1f}%",
            }
        )
    return rows


def get_current_stats(df: pd.DataFrame) -> dict:
    current_price = df["Close"].iloc[-1]
    prev_price = df["Close"].iloc[-2]
    change_pct = (current_price - prev_price) / prev_price * 100
    current_rsi = compute_rsi(df["Close"]).iloc[-1]

    today = pd.Timestamp.today()
    days_to_halving = (NEXT_HALVING - today).days
    h4_date = HALVINGS["H4 (2024)"]["date"]
    h4_price = HALVINGS["H4 (2024)"]["price"]

    return {
        "price": f"${current_price:,.0f}",
        "change": f"{change_pct:+.2f}%",
        "change_positive": change_pct >= 0,
        "rsi": f"{current_rsi:.1f}",
        "days_to_halving": f"{days_to_halving:,}",
        "days_since_h4": f"{(today - h4_date).days:,}",
        "h4_multiplier": f"{current_price / h4_price:.2f}x",
    }


# ── Chart builders ────────────────────────────────────────────────────────────
CYCLE_COLORS = ["#f59e0b", "#10b981", "#3b82f6", "#ec4899"]


def build_halving_chart(df: pd.DataFrame) -> go.Figure:
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

        # Shaded window
        fig.add_vrect(
            x0=start,
            x1=min(end, df.index[-1]),
            fillcolor=color,
            opacity=0.06,
            layer="below",
            line_width=0,
        )

        # Halving vertical line
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


def build_rsi_chart(df: pd.DataFrame) -> go.Figure:
    df = compute_emas(df)
    df["RSI"] = compute_rsi(df["Close"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
        subplot_titles=("Price & EMAs", "RSI (14)"),
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
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA_50"],
            mode="lines",
            name="EMA 50",
            line=dict(color="#10b981", width=1, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA_200"],
            mode="lines",
            name="EMA 200",
            line=dict(color="#3b82f6", width=1, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # RSI crossover signals
    buy = df[(df["RSI"].shift(1) < 30) & (df["RSI"] >= 30)]
    sell = df[(df["RSI"].shift(1) > 70) & (df["RSI"] <= 70)]

    fig.add_trace(
        go.Scatter(
            x=buy.index,
            y=buy["Close"],
            mode="markers",
            name="Buy Signal",
            marker=dict(color="#10b981", size=10, symbol="triangle-up"),
            hovertemplate="%{x|%Y-%m-%d}<br>Buy @ $%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sell.index,
            y=sell["Close"],
            mode="markers",
            name="Sell Signal",
            marker=dict(color="#ef4444", size=10, symbol="triangle-down"),
            hovertemplate="%{x|%Y-%m-%d}<br>Sell @ $%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            mode="lines",
            name="RSI",
            line=dict(color="#a78bfa", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>RSI: %{y:.1f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_hrect(y0=70, y1=100, fillcolor="#ef4444", opacity=0.05, layer="below", line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="#10b981", opacity=0.05, layer="below", line_width=0, row=2, col=1)
    fig.add_hline(y=70, line=dict(color="#ef4444", dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#10b981", dash="dash", width=1), row=2, col=1)

    fig.update_layout(
        title="RSI Strategy — Price, EMAs & Signals",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12),
        height=600,
        margin=dict(t=60, b=80),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    return fig


# ── UI helpers ────────────────────────────────────────────────────────────────
def stat_card(label: str, value: str, color: str = "#f59e0b") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(label, className="text-muted mb-1", style={"fontSize": "0.75rem", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                html.H5(value, style={"color": color, "fontWeight": "700", "marginBottom": 0}),
            ]
        ),
        className="h-100",
    )


TABLE_STYLE = dict(
    style_table={"overflowX": "auto"},
    style_header={
        "backgroundColor": "#1e293b",
        "color": "#f8fafc",
        "fontWeight": "bold",
        "border": "1px solid #334155",
        "textAlign": "center",
    },
    style_cell={
        "backgroundColor": "#0f172a",
        "color": "#e2e8f0",
        "border": "1px solid #1e293b",
        "padding": "10px 14px",
        "textAlign": "center",
        "fontFamily": "monospace",
    },
    style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#1e293b"}],
)


# ── Build layout (runs once at startup) ──────────────────────────────────────
_df = load_btc_data()
_stats = get_current_stats(_df)
_cycle_rows = get_cycle_analysis(_df)

app.layout = dbc.Container(
    [
        # ── Header ──────────────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Bitcoin Dashboard", className="mb-0 fw-bold"),
                        html.Small(
                            f"Live data via yfinance · updated {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M')} UTC",
                            className="text-muted",
                        ),
                    ],
                    width=12,
                    md=8,
                ),
            ],
            className="mt-4 mb-3",
        ),
        # ── Stat cards ──────────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(stat_card("Current Price", _stats["price"]), width=6, md=2, className="mb-3"),
                dbc.Col(
                    stat_card("24h Change", _stats["change"], "#10b981" if _stats["change_positive"] else "#ef4444"),
                    width=6,
                    md=2,
                    className="mb-3",
                ),
                dbc.Col(stat_card("RSI (14)", _stats["rsi"], "#a78bfa"), width=6, md=2, className="mb-3"),
                dbc.Col(stat_card("Days to H5", _stats["days_to_halving"], "#3b82f6"), width=6, md=2, className="mb-3"),
                dbc.Col(stat_card("Days Since H4", _stats["days_since_h4"], "#f59e0b"), width=6, md=2, className="mb-3"),
                dbc.Col(stat_card("H4 Multiplier", _stats["h4_multiplier"], "#ec4899"), width=6, md=2, className="mb-3"),
            ]
        ),
        # ── Halving chart ────────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dcc.Graph(
                            figure=build_halving_chart(_df),
                            config={"displayModeBar": True, "displaylogo": False},
                        )
                    )
                ),
                width=12,
            ),
            className="mb-4",
        ),
        # ── Cycle table ──────────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(
                [
                    html.H5("Halving Cycle Analysis", className="mb-3"),
                    dash_table.DataTable(
                        data=_cycle_rows,
                        columns=[{"name": c, "id": c} for c in _cycle_rows[0]],
                        **TABLE_STYLE,
                    ),
                ],
                width=12,
            ),
            className="mb-4",
        ),
        # ── RSI chart ────────────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dcc.Graph(
                            figure=build_rsi_chart(_df),
                            config={"displayModeBar": True, "displaylogo": False},
                        )
                    )
                ),
                width=12,
            ),
            className="mb-4",
        ),
        # ── Footer ───────────────────────────────────────────────────────────
        html.Hr(),
        html.P(
            "Data sourced from Yahoo Finance via yfinance. For educational purposes only.",
            className="text-muted text-center small mb-4",
        ),
    ],
    fluid=True,
)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
