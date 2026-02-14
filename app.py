"""
app.py — Dash server, layout, and callbacks.

──────────────────────────────────────────────────────────────────────────────
HOW TO ADD A NEW GENERIC CHART  (works for any ticker)
──────────────────────────────────────────────────────────────────────────────
1. Write `build_my_chart(df, ticker)` in charts.py
2. Add one entry to GENERIC_CHARTS below:
       {"id": "my-chart", "build": charts.build_my_chart},
3. Run `python app.py` — it appears automatically at http://localhost:8050

For BTC-specific charts, add them inside the `if is_btc:` block in
update_dashboard() below.
──────────────────────────────────────────────────────────────────────────────
"""

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import charts
from dash import Input, Output, dash_table, dcc, html
from data import (
    ASSET_CLASSES,
    DEFAULT_ASSET_CLASS,
    DEFAULT_TICKER,
    get_current_stats,
    get_cycle_analysis,
    load_ticker_data,
)

# ── App init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Finance Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # expose Flask server for Gunicorn / Render

# ── Generic chart registry ────────────────────────────────────────────────────
# Charts here must accept (df, ticker) and return a go.Figure.
# They are rendered for every selected ticker.
GENERIC_CHARTS = [
    {"id": "rsi-chart", "build": charts.build_rsi_chart},
    # Add new generic charts here ↓
]

# ── UI helpers ────────────────────────────────────────────────────────────────
_CARD_COL = dict(width=6, md=2, className="mb-3")

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

_DD_STYLE = {
    "backgroundColor": "#1e293b",
    "color": "#f8fafc",
    "border": "1px solid #334155",
}


def stat_card(label: str, value: str, color: str = "#f59e0b") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.P(
                    label,
                    className="text-muted mb-1",
                    style={"fontSize": "0.75rem", "textTransform": "uppercase", "letterSpacing": "0.05em"},
                ),
                html.H5(value, style={"color": color, "fontWeight": "700", "marginBottom": 0}),
            ]
        ),
        className="h-100",
    )


def make_stats_row(stats: dict) -> dbc.Row:
    """Build the top stat cards row. BTC gets halving-specific cards; others get 52W range."""
    base_cards = [
        dbc.Col(stat_card("Price", stats["price"]), **_CARD_COL),
        dbc.Col(
            stat_card("24h Change", stats["change"], "#10b981" if stats["change_positive"] else "#ef4444"),
            **_CARD_COL,
        ),
        dbc.Col(stat_card("RSI (14)", stats["rsi"], "#a78bfa"), **_CARD_COL),
    ]

    if stats["is_btc"]:
        extra_cards = [
            dbc.Col(stat_card("Days to H5", stats["days_to_halving"], "#3b82f6"), **_CARD_COL),
            dbc.Col(stat_card("Days Since H4", stats["days_since_h4"], "#f59e0b"), **_CARD_COL),
            dbc.Col(stat_card("H4 Multiplier", stats["h4_multiplier"], "#ec4899"), **_CARD_COL),
        ]
    else:
        extra_cards = [
            dbc.Col(stat_card("52W High", stats["high_52w"], "#10b981"), **_CARD_COL),
            dbc.Col(stat_card("52W Low", stats["low_52w"], "#ef4444"), **_CARD_COL),
        ]

    return dbc.Row(base_cards + extra_cards)


def make_cycle_table(rows: list[dict]) -> dash_table.DataTable:
    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in rows[0]],
        **TABLE_STYLE,
    )


# ── Layout ────────────────────────────────────────────────────────────────────
_initial_ticker_options = ASSET_CLASSES[DEFAULT_ASSET_CLASS]

app.layout = dbc.Container(
    [
        # ── Header ──────────────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(
                [
                    html.H2("Finance Dashboard", className="mb-0 fw-bold"),
                    html.Small("Live data via yfinance", className="text-muted"),
                ],
                width=12,
            ),
            className="mt-4 mb-3",
        ),

        # ── Filter row ───────────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Asset Class", className="text-muted small mb-1"),
                        dcc.Dropdown(
                            id="asset-class-dd",
                            options=[{"label": k, "value": k} for k in ASSET_CLASSES],
                            value=DEFAULT_ASSET_CLASS,
                            clearable=False,
                            style=_DD_STYLE,
                        ),
                    ],
                    width=6, md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Ticker", className="text-muted small mb-1"),
                        dcc.Dropdown(
                            id="ticker-dd",
                            options=_initial_ticker_options,
                            value=DEFAULT_TICKER,
                            clearable=False,
                            style=_DD_STYLE,
                        ),
                    ],
                    width=6, md=3,
                ),
            ],
            className="mb-4",
        ),

        # ── Dynamic content ──────────────────────────────────────────────────
        dcc.Loading(
            children=[
                # Stat cards (children replaced by callback)
                html.Div(id="stats-row", className="mb-3"),

                # BTC-specific section: cycle table + halving chart
                # Hidden for non-BTC tickers via style callback output
                html.Div(id="btc-section", style={"display": "none"}),

                # Generic charts — one dcc.Graph per entry in GENERIC_CHARTS
                *[
                    dbc.Row(
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    dcc.Graph(
                                        id=c["id"],
                                        config={"displayModeBar": True, "displaylogo": False},
                                    )
                                )
                            ),
                            width=12,
                        ),
                        className="mb-4",
                    )
                    for c in GENERIC_CHARTS
                ],
            ],
            color="#f59e0b",
            type="circle",
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


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("ticker-dd", "options"),
    Output("ticker-dd", "value"),
    Input("asset-class-dd", "value"),
)
def update_ticker_options(asset_class: str):
    """When asset class changes, repopulate the ticker dropdown and reset to first."""
    options = ASSET_CLASSES[asset_class]
    return options, options[0]["value"]


@app.callback(
    [
        Output("stats-row", "children"),
        Output("btc-section", "children"),
        Output("btc-section", "style"),
    ]
    + [Output(c["id"], "figure") for c in GENERIC_CHARTS],
    Input("ticker-dd", "value"),
)
def update_dashboard(ticker: str):
    """When ticker changes, refresh all charts, stat cards, and BTC-specific content."""
    if not ticker:
        empty_fig = go.Figure(layout=dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)"))
        return [html.Div(), [], {"display": "none"}] + [empty_fig] * len(GENERIC_CHARTS)

    df = load_ticker_data(ticker)
    is_btc = ticker == "BTC-USD"
    stats = get_current_stats(df, ticker)

    # Stat cards
    stats_children = make_stats_row(stats)

    # BTC-specific section
    if is_btc:
        cycle_rows = get_cycle_analysis(df)
        btc_children = [
            html.H5("Halving Cycle Analysis", className="mb-3 mt-2"),
            make_cycle_table(cycle_rows),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(
                                figure=charts.build_halving_chart(df),
                                config={"displayModeBar": True, "displaylogo": False},
                            )
                        )
                    ),
                    width=12,
                ),
                className="mt-4",
            ),
        ]
        btc_style = {"marginBottom": "1.5rem"}
    else:
        btc_children = []
        btc_style = {"display": "none"}

    # Generic charts (each function receives df + ticker)
    generic_figs = [c["build"](df, ticker) for c in GENERIC_CHARTS]

    return [stats_children, btc_children, btc_style] + generic_figs


if __name__ == "__main__":
    app.run(debug=True, port=8050)
