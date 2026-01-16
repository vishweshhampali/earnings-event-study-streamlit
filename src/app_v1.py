import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
from dotenv import load_dotenv
from pathlib import Path

# If you already have these in fmp_client.py / event_study.py, import them instead.
# For now, I’m importing from your existing logic style and keeping app.py self-contained.
import requests


# ------------------------------
# Config
# ------------------------------
st.set_page_config(page_title="FMP Earnings Event Study", layout="wide")

st.title("📊 Earnings Event Study (FMP)")
st.caption("Close prices only → event move (-w to +w), daily % change, CAR, and t-test significance.")


# ------------------------------
# Free plan allowed symbols (dropdown)
# ------------------------------
ALLOWED_SYMBOLS_FREE = [
    "AAPL","TSLA","AMZN","MSFT","NVDA","GOOGL","META","NFLX","JPM","V","BAC","PYPL","DIS","T","PFE",
    "COST","INTC","KO","TGT","NKE","SPY","BA","BABA","XOM","WMT","GE","CSCO","VZ","JNJ","CVX","PLTR",
    "SQ","SHOP","SBUX","SOFI","HOOD","RBLX","SNAP","AMD","UBER","FDX","ABBV","ETSY","MRNA","LMT","GM",
    "F","LCID","CCL","DAL","UAL","AAL","TSM","SONY","ET","MRO","COIN","RIVN","RIOT","CPRX","VWO","SPYG",
    "NOK","ROKU","VIAC","ATVI","BIDU","DOCU","ZM","PINS","TLRY","WBA","MGM","NIO","C","GS","WFC","ADBE",
    "PEP","UNH","CARR","HCA","TWTR","BILI","SIRI","FUBO","RKT"
]

# Fixed benchmark (no user input)
BENCH_SYMBOL = "SPY"


# ------------------------------
# API key helpers (Streamlit-friendly)
# ------------------------------

# Load .env from project root (one level above src/)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
API_KEY = os.getenv("FMP_API_KEY")
if not API_KEY:
    st.error("Missing FMP_API_KEY in .env")
    st.stop()



# ------------------------------
# Cached fetchers
# ------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60)  # 1 hour cache
def fetch_prices(symbol: str) -> pd.DataFrame:
    r = requests.get(
        "https://financialmodelingprep.com/stable/historical-price-eod/full",
        params={"symbol": symbol, "apikey": API_KEY},
        timeout=30
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")[["date", "close"]].reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_earnings(symbol: str) -> pd.DataFrame:
    r = requests.get(
        "https://financialmodelingprep.com/stable/earnings",
        params={"symbol": symbol, "apikey": API_KEY},
        timeout=30
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ------------------------------
# Event-study utilities (same as your backend logic) :contentReference[oaicite:1]{index=1}
# ------------------------------
def next_trading_pos(dates_array: np.ndarray, event_date) -> int | None:
    event_date = np.datetime64(pd.to_datetime(event_date))
    pos = dates_array.searchsorted(event_date)
    if pos >= len(dates_array):
        return None
    return int(pos)


def compute_abnormal_returns(
    stock_df: pd.DataFrame,
    market_df: pd.DataFrame,
    date_col: str = "date",
    close_col: str = "close",
) -> pd.DataFrame:
    stock = stock_df.copy().sort_values(date_col)
    market = market_df.copy().sort_values(date_col)

    stock["stock_ret"] = stock[close_col].pct_change()
    market["market_ret"] = market[close_col].pct_change()

    merged = stock.merge(market[[date_col, "market_ret"]], on=date_col, how="inner")
    merged["abnormal_ret"] = merged["stock_ret"] - merged["market_ret"]
    merged = merged.dropna(subset=["stock_ret", "market_ret", "abnormal_ret"])
    return merged


def compute_event_move_minus_w_to_plus_w(
    prices_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window: int,
    price_col: str = "close",
    event_col: str = "date",
) -> pd.DataFrame:
    """
    Chart 2 requirement:
    Plain move (%) from -w to +w for each earnings event.
    """
    prices = prices_df[["date", price_col]].copy().sort_values("date").reset_index(drop=True)
    events = events_df.copy()
    events[event_col] = pd.to_datetime(events[event_col])

    dates = prices["date"].to_numpy()
    closes = prices[price_col].to_numpy()

    rows = []
    dropped = 0

    for _, r in events.iterrows():
        ed = r[event_col]
        pos0 = next_trading_pos(dates, ed)
        if pos0 is None:
            dropped += 1
            continue

        pre_pos = pos0 - window
        post_pos = pos0 + window
        if pre_pos < 0 or post_pos >= len(dates):
            dropped += 1
            continue

        pre_close = closes[pre_pos]
        post_close = closes[post_pos]

        rows.append(
            {
                "earnings_date": ed,
                "event_trading_date": pd.to_datetime(dates[pos0]),
                f"move_-{window}_to_+{window}_pct": (post_close / pre_close - 1.0) * 100.0,
            }
        )

    out = pd.DataFrame(rows).sort_values("event_trading_date", ascending=True).reset_index(drop=True)
    out.attrs["dropped_events"] = dropped
    return out


def compute_car_per_event(
    merged_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window: int,
    date_col: str = "date",
    abn_col: str = "abnormal_ret",
    event_col: str = "date",
) -> pd.DataFrame:
    """
    Chart 4 requirement:
    CAR (%) summed from -w to +w for each earnings event.
    """
    m = merged_df[[date_col, abn_col]].copy().sort_values(date_col).reset_index(drop=True)
    e = events_df.copy()
    e[event_col] = pd.to_datetime(e[event_col])

    dates = m[date_col].to_numpy()
    abn = m[abn_col].to_numpy()

    rows = []
    dropped = 0

    for ed in e[event_col]:
        pos0 = next_trading_pos(dates, ed)
        if pos0 is None:
            dropped += 1
            continue

        pre_pos = pos0 - window
        post_pos = pos0 + window
        if pre_pos < 0 or post_pos >= len(dates):
            dropped += 1
            continue

        car = np.nansum(abn[pre_pos : post_pos + 1]) * 100.0
        rows.append(
            {
                "earnings_date": pd.to_datetime(ed),
                "event_trading_date": pd.to_datetime(dates[pos0]),
                f"CAR_-{window}_to_+{window}_pct": car,
            }
        )

    out = pd.DataFrame(rows).sort_values("event_trading_date", ascending=True).reset_index(drop=True)
    out.attrs["dropped_events"] = dropped
    return out


# ------------------------------
# Sidebar inputs
# ------------------------------
with st.sidebar:
    st.header("Inputs")

    stock_symbol = st.selectbox(
        "Stock symbol (Free plan)",
        options=ALLOWED_SYMBOLS_FREE,
        index=ALLOWED_SYMBOLS_FREE.index("AAPL") if "AAPL" in ALLOWED_SYMBOLS_FREE else 0
    )

    st.caption(f"Benchmark is fixed to: **{BENCH_SYMBOL}**")

    window = st.slider("Window (trading days)", 1, 5, 3)
    run = st.button("Run analysis", type="primary")



if not run:
    st.info("Set inputs on the left and click **Run analysis**.")
    st.stop()


# ------------------------------
# Fetch data
# ------------------------------
with st.spinner("Fetching prices & earnings..."):
    stock_px = fetch_prices(stock_symbol)
    bench_px = fetch_prices(BENCH_SYMBOL)
    earnings = fetch_earnings(stock_symbol)

# ------------------------------
# Anchor the analysis range to STOCK availability
# ------------------------------
stock_min = stock_px["date"].min()
stock_max = stock_px["date"].max()

# Keep benchmark only where stock exists
bench_px = bench_px[(bench_px["date"] >= stock_min) & (bench_px["date"] <= stock_max)].reset_index(drop=True)
if bench_px.empty:
    st.error(f"{BENCH_SYMBOL} has no data within {stock_symbol}'s date range ({stock_min.date()} to {stock_max.date()}).")
    st.stop()

# Final usable range (stock anchored, but also respects benchmark end)
range_min = stock_min
range_max = min(stock_max, bench_px["date"].max())

stock_px = stock_px[(stock_px["date"] >= range_min) & (stock_px["date"] <= range_max)].reset_index(drop=True)
bench_px = bench_px[(bench_px["date"] >= range_min) & (bench_px["date"] <= range_max)].reset_index(drop=True)


# Keep only earnings dates that fall within the usable analysis range
earnings = earnings[(earnings["date"] >= range_min) & (earnings["date"] <= range_max)].reset_index(drop=True)

# Optional safety cap (not a UI input) to keep charts readable and fast.
# Comment this out if you truly want ALL events.
MAX_EVENTS = 40
if len(earnings) > MAX_EVENTS:
    earnings = earnings.tail(MAX_EVENTS).reset_index(drop=True)

if earnings.empty:
    st.error("No earnings dates found within the available price history.")
    st.stop()


# ------------------------------
# Chart 1: Close prices (stock + benchmark) + earnings markers
# ------------------------------
st.subheader("1) Close prices (Stock vs Benchmark)")

# Align on common dates
px = stock_px.merge(bench_px, on="date", how="inner", suffixes=(f"_{stock_symbol}", f"_{BENCH_SYMBOL}"))

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=px["date"],
    y=px[f"close_{stock_symbol}"],
    mode="lines",
    name=f"{stock_symbol} Close"
))
fig1.add_trace(go.Scatter(
    x=px["date"],
    y=px[f"close_{BENCH_SYMBOL}"],
    mode="lines",
    name=f"{BENCH_SYMBOL} Close"
))

# Earnings markers
for d in earnings["date"]:
    fig1.add_vline(x=d, line_width=1, opacity=0.25)

fig1.update_layout(
    height=450,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(title="Close price"),
    legend=dict(orientation="h"),
)
st.plotly_chart(fig1, use_container_width=True)


# ------------------------------
# Chart 3: Daily % change (full timeline)
# ------------------------------
st.subheader("2) Daily % change (today vs yesterday)")

# Daily pct change from close prices
stock_ret = stock_px.copy()
bench_ret = bench_px.copy()
stock_ret["daily_pct"] = stock_ret["close"].pct_change() * 100.0
bench_ret["daily_pct"] = bench_ret["close"].pct_change() * 100.0

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=stock_ret["date"], y=stock_ret["daily_pct"], mode="lines", name=f"{stock_symbol} Daily %"))
fig3.add_trace(go.Scatter(x=bench_ret["date"], y=bench_ret["daily_pct"], mode="lines", name=f"{BENCH_SYMBOL} Daily %"))
fig3.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(title="Daily % change"),
    legend=dict(orientation="h"),
)
st.plotly_chart(fig3, use_container_width=True)


# ------------------------------
# Chart 2: Plain move from -w to +w per event (event dates on X)
# ------------------------------
st.subheader(f"3) Plain move from -{window} to +{window} (by earnings event)")

stock_move = compute_event_move_minus_w_to_plus_w(stock_px, earnings, window=window)
bench_move = compute_event_move_minus_w_to_plus_w(bench_px, earnings, window=window)

# Join on event_trading_date for clean plotting
move_col = f"move_-{window}_to_+{window}_pct"
plot_move = stock_move[["event_trading_date", move_col]].merge(
    bench_move[["event_trading_date", move_col]],
    on="event_trading_date",
    how="inner",
    suffixes=(f"_{stock_symbol}", f"_{BENCH_SYMBOL}"),
)

dropped_msg = []
if stock_move.attrs.get("dropped_events", 0) > 0:
    dropped_msg.append(f"{stock_symbol} dropped: {stock_move.attrs['dropped_events']}")
if bench_move.attrs.get("dropped_events", 0) > 0:
    dropped_msg.append(f"{BENCH_SYMBOL} dropped: {bench_move.attrs['dropped_events']}")
if dropped_msg:
    st.warning("Some events were dropped due to insufficient window data. " + " | ".join(dropped_msg))

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=plot_move["event_trading_date"], y=plot_move[f"{move_col}_{stock_symbol}"], name=f"{stock_symbol} Move (%)"))
fig2.add_trace(go.Bar(x=plot_move["event_trading_date"], y=plot_move[f"{move_col}_{BENCH_SYMBOL}"], name=f"{BENCH_SYMBOL} Move (%)"))
fig2.update_layout(
    barmode="group",
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(title=f"% move from -{window} to +{window}"),
    legend=dict(orientation="h"),
)
st.plotly_chart(fig2, use_container_width=True)


# ------------------------------
# Chart 4: CAR per event (event dates on X) + t-test
# ------------------------------
st.subheader(f"4) CAR from -{window} to +{window} (by earnings event)")

merged = compute_abnormal_returns(stock_px, bench_px)
car_df = compute_car_per_event(merged, earnings, window=window)
car_col = f"CAR_-{window}_to_+{window}_pct"

if car_df.empty:
    st.error("No CAR values could be computed (likely due to insufficient window history). Try a smaller window.")
    st.stop()

fig4 = go.Figure()
fig4.add_trace(go.Bar(x=car_df["event_trading_date"], y=car_df[car_col], name="CAR (%)"))
fig4.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(title="CAR (%)"),
)
st.plotly_chart(fig4, use_container_width=True)

# Hypothesis + t-test
st.subheader("Hypothesis test (1-sample t-test on CAR)")

car_sample = car_df[car_col].dropna().astype(float)
n = len(car_sample)

if n < 2:
    st.warning("Not enough CAR samples to run a t-test. Increase earnings count or reduce window.")
    st.stop()

mean_car = float(car_sample.mean())
std_car = float(car_sample.std(ddof=1))

t_stat, p_value = stats.ttest_1samp(car_sample, popmean=0.0)

# 95% CI for mean CAR
alpha = 0.05
dfree = n - 1
t_crit = stats.t.ppf(1 - alpha / 2, dfree)
se = std_car / np.sqrt(n)
ci_low = mean_car - t_crit * se
ci_high = mean_car + t_crit * se

c1, c2, c3, c4 = st.columns(4)
c1.metric("N events", f"{n}")
c2.metric("Mean CAR (%)", f"{mean_car:.3f}")
c3.metric("t-stat", f"{t_stat:.3f}")
c4.metric("p-value", f"{p_value:.4f}")

st.markdown(
    f"""
**H0:** Mean CAR = 0  
**H1:** Mean CAR ≠ 0  

95% CI for mean CAR: **[{ci_low:.3f}, {ci_high:.3f}]**  
"""
)

if p_value < alpha:
    st.success(f"Conclusion: Reject H0 at α={alpha}. Average CAR is statistically different from 0.")
else:
    st.info(f"Conclusion: Fail to reject H0 at α={alpha}. Not enough evidence that average CAR differs from 0.")
