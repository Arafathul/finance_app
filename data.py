"""
data.py — data loading, caching, and computed analytics.

All chart builders in charts.py receive a DataFrame from load_ticker_data().
Add new derived columns or analytics functions here.
"""
import time

import numpy as np
import pandas as pd
import yfinance as yf

# ── Asset class / ticker catalogue ───────────────────────────────────────────
# Each entry is a list of {"label": "...", "value": "TICKER"} dicts for dropdowns.
ASSET_CLASSES: dict[str, list[dict]] = {
    "Crypto": [
        {"label": "Bitcoin (BTC)",   "value": "BTC-USD"},
        {"label": "Ethereum (ETH)",  "value": "ETH-USD"},
        {"label": "Solana (SOL)",    "value": "SOL-USD"},
        {"label": "BNB (BNB)",       "value": "BNB-USD"},
        {"label": "XRP (XRP)",       "value": "XRP-USD"},
        {"label": "Dogecoin (DOGE)", "value": "DOGE-USD"},
        {"label": "Cardano (ADA)",    "value": "ADA-USD"},
        {"label": "Polkadot (DOT)",   "value": "DOT-USD"},
        {"label": "Avalanche (AVAX)", "value": "AVAX-USD"},
        {"label":"Algorand (ALGO)",     "value": "ALGO-USD"},
        {"label":"Cosmos (ATOM)",        "value": "ATOM-USD"},
        {"label":"Chainlink (LINK)",     "value": "LINK-USD"},
        {"label":"Hedera (HBAR)",        "value": "HBAR-USD"},
         
    ],
    "Stocks": [
        {"label": "Apple (AAPL)",    "value": "AAPL"},
        {"label": "Microsoft (MSFT)","value": "MSFT"},
        {"label": "NVIDIA (NVDA)",   "value": "NVDA"},
        {"label": "Tesla (TSLA)",    "value": "TSLA"},
        {"label": "Amazon (AMZN)",   "value": "AMZN"},
        {"label": "Google (GOOGL)",  "value": "GOOGL"},
        {"label": "Meta (META)",     "value": "META"},
    ],
    "ETFs": [
        {"label": "S&P 500 (SPY)",        "value": "SPY"},
        {"label": "NASDAQ 100 (QQQ)",     "value": "QQQ"},
        {"label": "Total Market (VTI)",   "value": "VTI"},
        {"label": "Gold (GLD)",           "value": "GLD"},
        {"label": "Long Bonds (TLT)",     "value": "TLT"},
    ],
    "Forex": [
        {"label": "EUR/USD", "value": "EURUSD=X"},
        {"label": "GBP/USD", "value": "GBPUSD=X"},
        {"label": "USD/JPY", "value": "JPY=X"},
        {"label": "AUD/USD", "value": "AUDUSD=X"},
        {"label": "USD/CAD", "value": "CAD=X"},
    ],
    "Bonds": [
        {"label": "10Y Treasury (^TNX)", "value": "^TNX"},
        {"label": "30Y Treasury (^TYX)", "value": "^TYX"},
        {"label": "2Y Treasury (^IRX)",  "value": "^IRX"},
    ],
}

DEFAULT_ASSET_CLASS = "Crypto"
DEFAULT_TICKER = "BTC-USD"

# ── BTC halving constants ─────────────────────────────────────────────────────
HALVINGS = {
    "H1 (2012)": {"date": pd.Timestamp("2012-11-28"), "price": 12.35},
    "H2 (2016)": {"date": pd.Timestamp("2016-07-09"), "price": 650.63},
    "H3 (2020)": {"date": pd.Timestamp("2020-05-11"), "price": 8_821.42},
    "H4 (2024)": {"date": pd.Timestamp("2024-04-20"), "price": 63_850.00},
}
NEXT_HALVING = pd.Timestamp("2028-04-17")
BUFFER_DAYS = 60
CACHE_TTL = 3_600  # seconds

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: dict = {}
_cache_times: dict = {}


def load_ticker_data(ticker: str) -> pd.DataFrame:
    """Fetch daily price history for any yfinance ticker, cached for 1 hour."""
    now = time.time()
    if ticker in _cache and now - _cache_times.get(ticker, 0) < CACHE_TTL:
        return _cache[ticker]
    df = yf.Ticker(ticker).history(period="max", interval="1d")
    df.index = df.index.tz_localize(None)
    _cache[ticker] = df
    _cache_times[ticker] = now
    return df


# ── Indicators ────────────────────────────────────────────────────────────────
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
    """Return per-halving cycle stats: peak, low, days, multiplier, drawdown. BTC only."""
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


def get_current_stats(df: pd.DataFrame, ticker: str = DEFAULT_TICKER) -> dict:
    """
    Return formatted stat card values.

    All tickers: price, 24h change, RSI, 52W high/low.
    BTC only:    days_to_halving, days_since_h4, h4_multiplier.
    """
    current_price = df["Close"].iloc[-1]
    prev_price = df["Close"].iloc[-2]
    change_pct = (current_price - prev_price) / prev_price * 100
    current_rsi = compute_rsi(df["Close"]).iloc[-1]
    high_52w = df["Close"].tail(252).max()
    low_52w = df["Close"].tail(252).min()
    is_btc = ticker == "BTC-USD"

    stats: dict = {
        "price": f"${current_price:,.2f}",
        "change": f"{change_pct:+.2f}%",
        "change_positive": change_pct >= 0,
        "rsi": f"{current_rsi:.1f}",
        "high_52w": f"${high_52w:,.2f}",
        "low_52w": f"${low_52w:,.2f}",
        "is_btc": is_btc,
    }

    if is_btc:
        today = pd.Timestamp.today()
        h4_date = HALVINGS["H4 (2024)"]["date"]
        h4_price = HALVINGS["H4 (2024)"]["price"]
        stats.update(
            {
                "days_to_halving": f"{(NEXT_HALVING - today).days:,}",
                "days_since_h4": f"{(today - h4_date).days:,}",
                "h4_multiplier": f"{current_price / h4_price:.2f}x",
            }
        )

    return stats
