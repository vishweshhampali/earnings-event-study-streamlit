import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
from dotenv import load_dotenv
from pathlib import Path
import requests


# ------------------------------
# Config
# ------------------------------
st.set_page_config(page_title="FMP Earnings Event Study", layout="wide")

st.title("📊 Returns around Earnings Event")
#st.caption("Close prices only → event move (-w to +w), daily % change, CAR, and t-test significance.")


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
# Event-study utilities (same as your backend logic)
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
    st.caption(f"Window = **−{window} to +{window} trading days** around each earnings date.")

    run = st.button("Run analysis", type="primary")

    st.divider()
    st.caption("**Did you know?**")
    st.markdown(
        """
        The *t-test* was introduced by **William Sealy Gosset** (working at Guinness Brewery).  
        He published under the pen name **“Student”** — that’s why it’s often called *Student’s t-test*.
        """
    )


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
# SECTION 0 — Core question
# ------------------------------
st.header("The Question We’re Testing")

st.markdown(
    """
**Do earnings announcements create a short-term trading edge?**  
>**If I repeatedly buy before earnings and sell shortly after, would I expect to make money on average?**

Earnings can trigger big reactions, but big reactions don’t automatically mean opportunity.  
This analysis checks whether the **average earnings-window return** is **meaningfully different from zero**, across a set of earnings events.
"""
)

st.caption(
    "Goal: quantify the typical earnings-window return (CAR - Cumulative Abnormal Return) and check whether it’s reliably different from 0 across many earnings events."
)
st.divider()

# ------------------------------
# SECTION 1 — Data credibility
# ------------------------------
st.header("The Data")

st.markdown(
    """
### What data goes into this analysis?

This study is built on **three core inputs**:

- **Daily adjusted closing prices** for the selected stock  
- **Daily adjusted closing prices** for a benchmark index (**SPY**)  
- **Earnings announcement dates** for the selected stock  

All data is fetched programmatically from the  
**Financial Modeling Prep (FMP) API**.
"""
)

with st.expander("View input data samples (first 10 rows)"):
    c1, c2, c3 = st.columns(3)

    c1.write(f"**{stock_symbol} — daily prices**")
    c1.dataframe(stock_px.head(10), use_container_width=True)

    c2.write(f"**{BENCH_SYMBOL} — daily prices**")
    c2.dataframe(bench_px.head(10), use_container_width=True)

    c3.write("**Earnings dates**")
    c3.dataframe(earnings.head(10), use_container_width=True)


# Chart 1: Close prices (stock + benchmark) + earnings markers
st.subheader("Daily closing prices — Stock vs Benchmark")


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
st.divider()


# ------------------------------
# SECTION 2 — Event-level behavior
# ------------------------------
st.header("What Happens Around Earnings")

st.markdown(
    f"""
In this section we look at **how prices behave around each earnings event** using a ±{window}-day trading window.

We show two views:
- **Plain move (%):** the stock’s raw price move from **−{window} to +{window}**
- **CAR (Cumulative Abnormal Return)(%):** the same window, but adjusted for the market (**stock return − SPY return**)
"""
)

st.info("“Earnings create volatility, but not consistent direction.”")

# ---- Chart A: Plain move from -w to +w (Stock vs Benchmark) ----
st.subheader(f"How much does the stock move around earnings? (-{window} to +{window})")

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
fig2.add_trace(go.Bar(
    x=plot_move["event_trading_date"],
    y=plot_move[f"{move_col}_{stock_symbol}"],
    name=f"{stock_symbol} Move (%)"
))
fig2.add_trace(go.Bar(
    x=plot_move["event_trading_date"],
    y=plot_move[f"{move_col}_{BENCH_SYMBOL}"],
    name=f"{BENCH_SYMBOL} Move (%)"
))
fig2.update_layout(
    barmode="group",
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(title=f"% move from -{window} to +{window}"),
    legend=dict(orientation="h"),
)
st.plotly_chart(fig2, use_container_width=True)

# ---- Chart B: CAR per event ----
st.subheader(f"Does the move remain after adjusting for the market? (CAR adjusted with {BENCH_SYMBOL} returns)")

merged = compute_abnormal_returns(stock_px, bench_px)
car_df = compute_car_per_event(merged, earnings, window=window)
car_col = f"CAR_-{window}_to_+{window}_pct"

if car_df.empty:
    st.error("No CAR values could be computed (likely due to insufficient window history). Try a smaller window.")
    st.stop()

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=car_df["event_trading_date"],
    y=car_df[car_col],
    name="CAR (%)"
))
fig4.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(title="CAR (%)"),
)
st.plotly_chart(fig4, use_container_width=True)

# ---- Expander: show the exact data behind the charts ----
with st.expander("View how the chart values are computed (event-level details)"):
    st.markdown("### Chart A — The prices behind the move (-w vs +w)")

    # --- Build a detailed move table with -w and +w closes ---
    stock_dates = stock_px["date"].to_numpy()
    stock_closes = stock_px["close"].to_numpy()

    bench_dates = bench_px["date"].to_numpy()
    bench_closes = bench_px["close"].to_numpy()

    move_rows = []
    for ed in earnings["date"]:
        # Align event to next trading day in each series
        pos_s = next_trading_pos(stock_dates, ed)
        pos_b = next_trading_pos(bench_dates, ed)
        if pos_s is None or pos_b is None:
            continue

        pre_s, post_s = pos_s - window, pos_s + window
        pre_b, post_b = pos_b - window, pos_b + window

        if pre_s < 0 or post_s >= len(stock_dates):
            continue
        if pre_b < 0 or post_b >= len(bench_dates):
            continue

        s_pre_close = float(stock_closes[pre_s])
        s_post_close = float(stock_closes[post_s])
        s_move = (s_post_close / s_pre_close - 1.0) * 100.0

        b_pre_close = float(bench_closes[pre_b])
        b_post_close = float(bench_closes[post_b])
        b_move = (b_post_close / b_pre_close - 1.0) * 100.0

        move_rows.append({
            "earnings_date": pd.to_datetime(ed),
            "event_trading_date": pd.to_datetime(stock_dates[pos_s]),
            f"{stock_symbol} close (-{window})": s_pre_close,
            f"{stock_symbol} close (+{window})": s_post_close,
            f"{stock_symbol} move % (-{window}→+{window})": s_move,
            f"{BENCH_SYMBOL} close (-{window})": b_pre_close,
            f"{BENCH_SYMBOL} close (+{window})": b_post_close,
            f"{BENCH_SYMBOL} move % (-{window}→+{window})": b_move,
        })

    move_detail_df = pd.DataFrame(move_rows).sort_values("event_trading_date").reset_index(drop=True)
    st.dataframe(move_detail_df, use_container_width=True)

    st.divider()
    st.markdown("### Chart B — How CAR is built inside the window (stock vs market)")

    # merged already has stock_ret, market_ret, abnormal_ret by date
    # car_df has event_trading_date and CAR column (car_col)
    event_options = car_df["event_trading_date"].dt.strftime("%Y-%m-%d").tolist()
    selected = st.selectbox("Pick an earnings event (event trading date)", options=event_options, index=len(event_options)-1)

    selected_date = pd.to_datetime(selected)
    dates_m = merged["date"].to_numpy()
    pos0 = next_trading_pos(dates_m, selected_date)

    if pos0 is None:
        st.warning("Could not locate this event in the merged returns table.")
    else:
        pre_pos = pos0 - window
        post_pos = pos0 + window

        if pre_pos < 0 or post_pos >= len(merged):
            st.warning("Not enough data around this event for the selected window.")
        else:
            window_df = merged.iloc[pre_pos:post_pos+1].copy()

            # Make the CAR calculation explicit
            window_df["abnormal_ret_pct"] = window_df["abnormal_ret"] * 100.0
            window_df["cum_abnormal_ret_pct"] = window_df["abnormal_ret_pct"].cumsum()

            show_cols = [
                "date",
                "stock_ret",
                "market_ret",
                "abnormal_ret",
                "abnormal_ret_pct",
                "cum_abnormal_ret_pct",
            ]
            st.dataframe(window_df[show_cols], use_container_width=True)

            computed_car = float(window_df["abnormal_ret"].sum() * 100.0)
            st.metric(f"Computed CAR (-{window}→+{window}) %", f"{computed_car:.3f}")

            # (Optional) sanity check vs your car_df value for the same event date
            match = car_df[car_df["event_trading_date"] == selected_date]
            if not match.empty:
                st.caption(f"CAR shown in the chart/table for this event: **{float(match[car_col].iloc[0]):.3f}%**")

st.divider()

# ------------------------------
# SECTION 3 — Statistical validation
# ------------------------------
st.header("Is the Pattern Real or Just Luck?")

st.markdown(
    """
### Situation

Earnings windows can produce big moves, but direction varies by event.

You might **get lucky once**. The question is whether that result is **repeatable**.

> **Is there a repeatable average effect across earnings events?**

If there’s a real edge, the **average CAR** across many events should be **consistently above (or below) zero**.
"""
)


# Hypothesis + t-test
st.subheader("Hypothesis test (One-sample t-test on CAR)")

car_sample = car_df[car_col].dropna().astype(float)
n = len(car_sample)

if n < 2:
    st.warning("Not enough CAR samples to run a t-test. Increase earnings count or reduce window.")
    st.stop()

mean_car = float(car_sample.mean())
std_car = float(car_sample.std(ddof=1))

st.markdown(
    f"""
### Test setup (what we have)

- **Sample:** CAR values from **{n}** earnings events  
- **What we compare against:** a “no edge” baseline, where the **true mean CAR = 0%**  
- **Why 0%?** If earnings don’t provide a systematic advantage, the long-run average abnormal return in this window should be **approximately zero**.

### Hypotheses

- **H0 (null):** Mean CAR = 0  
- **H1 (alternative):** Mean CAR ≠ 0  *(two-sided test)*

### Why a one-sample t-test?

We have **one sample** of CAR values (one per earnings event) and we want to test whether its mean differs from a fixed reference value (0).  
A t-test is appropriate because the **population standard deviation is unknown** and we estimate variability from the sample.
"""
)

# ---- Run test ----
t_stat, p_value = stats.ttest_1samp(car_sample, popmean=0.0)

# 95% CI for mean CAR
alpha = 0.05
dfree = n - 1
t_crit = stats.t.ppf(1 - alpha / 2, dfree)
se = std_car / np.sqrt(n)
ci_low = mean_car - t_crit * se
ci_high = mean_car + t_crit * se

st.markdown("### What the test says")

m1, m2, m3, m4 = st.columns(4)
m1.metric("N events", f"{n}")
m2.metric("Mean CAR (%)", f"{mean_car:.3f}")
m3.metric("t-stat", f"{t_stat:.3f}")
m4.metric("p-value", f"{p_value:.4f}")

st.markdown(
    f"""
- **95% confidence interval for mean CAR:** **[{ci_low:.3f}, {ci_high:.3f}]**
- **Decision rule:** if **p-value < {alpha}**, we reject H0.
"""
)

if p_value < alpha:
    st.success(
        f"Conclusion: Reject H0 at α={alpha}. "
        "The average CAR is statistically different from 0 for this sample."
    )
else:
    st.info(
        f"Conclusion: Fail to reject H0 at α={alpha}. "
        "We don’t have enough evidence that the average CAR differs from 0 — "
        "any apparent edge could be due to noise."
    )

st.divider()

# ------------------------------
# SECTION 4 — Conclusion
# ------------------------------
st.header("Conclusion")

# A simple narrative summary based on the computed stats (no new variables needed, just reuse above)
if p_value < alpha:
    st.success(
        "High volatility *and* evidence of a systematic average effect in this window (mean CAR significantly different from 0). "
        "If this persists across more events / stocks, it may suggest a repeatable edge — but validate on more samples."
    )
else:
    st.info(
        "High volatility, **no reliable short-term edge**: event-window returns swing both ways, "
        "and the average CAR is not statistically different from zero for this sample."
    )

st.caption("Tip: Try different windows (1–5 days) and compare how the CAR distribution and test results change.")
