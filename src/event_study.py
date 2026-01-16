# src/event_study.py
import numpy as np
import pandas as pd
from scipy import stats

def compute_abnormal_returns(
    stock_df: pd.DataFrame,
    market_df: pd.DataFrame,
    date_col: str = "date",
    close_col: str = "close",
    stock_ret_col: str = "stock_ret",
    market_ret_col: str = "market_ret",
    abn_col: str = "abnormal_ret",
    dropna: bool = True,
) -> pd.DataFrame:
    stock = stock_df.copy().sort_values(date_col)
    market = market_df.copy().sort_values(date_col)

    stock[stock_ret_col] = stock[close_col].pct_change()
    market[market_ret_col] = market[close_col].pct_change()

    merged = stock.merge(market[[date_col, market_ret_col]], on=date_col, how="inner")
    merged[abn_col] = merged[stock_ret_col] - merged[market_ret_col]

    if dropna:
        merged = merged.dropna(subset=[stock_ret_col, market_ret_col, abn_col])

    return merged

def next_trading_pos(dates_array, event_date):
    event_date = np.datetime64(pd.to_datetime(event_date))
    pos = dates_array.searchsorted(event_date)
    if pos >= len(dates_array):
        return None
    return int(pos)

def compute_event_window_returns(prices_df, events_df, window=3, price_col="close", event_col="date"):
    prices = prices_df[["date", price_col]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").reset_index(drop=True)

    events = events_df.copy()
    events[event_col] = pd.to_datetime(events[event_col])

    dates = prices["date"].to_numpy()
    closes = prices[price_col].to_numpy()

    rows = []
    for _, r in events.iterrows():
        ed = r[event_col]
        pos0 = next_trading_pos(dates, ed)
        if pos0 is None:
            continue

        pre_pos = pos0 - window
        post_pos = pos0 + window
        if pre_pos < 0 or post_pos >= len(dates):
            continue

        pre_close = closes[pre_pos]
        post_close = closes[post_pos]
        event_close = closes[pos0]
        prev_close = closes[pos0 - 1] if pos0 - 1 >= 0 else np.nan

        rows.append({
            "earnings_date": ed,
            "event_trading_date": pd.to_datetime(dates[pos0]),
            f"pct_move_-{window}_to_+{window}": (post_close / pre_close - 1) * 100,
            "pct_move_day0_vs_prevclose": (event_close / prev_close - 1) * 100 if pd.notna(prev_close) else np.nan,
        })

    return pd.DataFrame(rows).sort_values("event_trading_date", ascending=False).reset_index(drop=True)

def compute_car(merged_df, events_df, window=3, date_col="date", abn_col="abnormal_ret", event_col="date"):
    m = merged_df[[date_col, abn_col]].copy()
    m[date_col] = pd.to_datetime(m[date_col])
    m = m.sort_values(date_col).reset_index(drop=True)

    e = events_df.copy()
    e[event_col] = pd.to_datetime(e[event_col])

    dates = m[date_col].to_numpy()
    abn = m[abn_col].to_numpy()

    rows = []
    for ed in e[event_col]:
        pos0 = next_trading_pos(dates, ed)
        if pos0 is None:
            continue

        pre_pos = pos0 - window
        post_pos = pos0 + window
        if pre_pos < 0 or post_pos >= len(dates):
            continue

        car = np.nansum(abn[pre_pos:post_pos + 1]) * 100

        rows.append({
            "earnings_date": pd.to_datetime(ed),
            "event_trading_date": pd.to_datetime(dates[pos0]),
            f"CAR_-{window}_to_+{window}_%": car
        })

    return pd.DataFrame(rows).sort_values("event_trading_date", ascending=False).reset_index(drop=True)

def summarize_and_test_car(car_df: pd.DataFrame, car_col: str, seed: int = 42, B: int = 10000):
    car_sample = car_df[car_col].dropna().astype(float)
    n = len(car_sample)

    if n < 3:
        return {
            "n": n,
            "error": "Not enough events to run statistical tests. Try a larger price history or different symbol."
        }

    mean_car = car_sample.mean()
    median_car = car_sample.median()
    std_car = car_sample.std(ddof=1)

    t_stat, p_value = stats.ttest_1samp(car_sample, popmean=0.0)

    alpha = 0.05
    dfree = n - 1
    t_crit = stats.t.ppf(1 - alpha/2, dfree)

    se = std_car / np.sqrt(n)
    ci_low = mean_car - t_crit * se
    ci_high = mean_car + t_crit * se

    rng = np.random.default_rng(seed)
    car_vals = car_sample.to_numpy()
    boot_means = np.empty(B)
    for i in range(B):
        sample = rng.choice(car_vals, size=n, replace=True)
        boot_means[i] = sample.mean()
    boot_low, boot_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "n": n,
        "mean_car": float(mean_car),
        "median_car": float(median_car),
        "std_car": float(std_car),
        "pct_positive": float((car_sample > 0).mean() * 100),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "boot_mean": float(boot_means.mean()),
        "boot_low": float(boot_low),
        "boot_high": float(boot_high),
        "car_sample": car_sample,  # keep for charts
    }


def build_event_panel(
    merged_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window: int = 3,
    date_col: str = "date",
    event_col: str = "date",
    stock_ret_col: str = "stock_ret",
    market_ret_col: str = "market_ret",
    abn_col: str = "abnormal_ret",
) -> pd.DataFrame:
    """
    Creates an event-panel of daily returns around each event:
    rel_day in [-window, +window], for every event that has full coverage.
    """
    m = merged_df.copy()
    m[date_col] = pd.to_datetime(m[date_col])
    m = m.sort_values(date_col).reset_index(drop=True)

    e = events_df.copy()
    e[event_col] = pd.to_datetime(e[event_col])

    dates = m[date_col].to_numpy()

    rows = []
    for ed in e[event_col]:
        # find same or next trading day for the event
        pos0 = next_trading_pos(dates, ed)
        if pos0 is None:
            continue

        pre_pos = pos0 - window
        post_pos = pos0 + window
        if pre_pos < 0 or post_pos >= len(m):
            continue

        slice_df = m.iloc[pre_pos:post_pos + 1].copy()
        slice_df["rel_day"] = np.arange(-window, window + 1, dtype=int)
        slice_df["event_trading_date"] = pd.to_datetime(dates[pos0])

        rows.append(slice_df[[ "event_trading_date", "rel_day",
                               stock_ret_col, market_ret_col, abn_col ]])

    if not rows:
        return pd.DataFrame(columns=["event_trading_date", "rel_day", stock_ret_col, market_ret_col, abn_col])

    return pd.concat(rows, ignore_index=True)

def summarize_event_panel(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average daily returns across events for each rel_day.
    Also computes mean CAR-by-day (cumulative abnormal return across rel_day).
    """
    if panel_df.empty:
        return pd.DataFrame(columns=["rel_day", "mean_stock_ret", "mean_market_ret", "mean_abnormal_ret", "mean_CAR_%"])

    g = panel_df.groupby("rel_day", as_index=False).agg(
        mean_stock_ret=("stock_ret", "mean"),
        mean_market_ret=("market_ret", "mean"),
        mean_abnormal_ret=("abnormal_ret", "mean"),
    )

    g = g.sort_values("rel_day").reset_index(drop=True)

    # Mean CAR by day = cumulative sum of mean abnormal returns
    g["mean_CAR_%"] = (g["mean_abnormal_ret"].cumsum() * 100.0)

    return g