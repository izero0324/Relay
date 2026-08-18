"""
backtest.py — Historical simulation of the Hold/Switch rotation strategy.
════════════════════════════════════════════════════════════════════════════

Simulates the daily decision loop on real historical OHLCV data.

Methodology
───────────
  • Signals computed at close of day t-1  (no lookahead)
  • Trades execute at the OPEN of day t (you can't trade yesterday's close):
      - exits earn the overnight move close[t-1] → open[t] on the old position
      - entries earn open[t] → close[t] on the new position
  • Default signals: Momentum + Volume. `--include-event-flow` adds
    point-in-time OHLCV/market proxies; unavailable historical Yahoo news and
    pre-market snapshots are not silently backfilled with present-day data.
  • Transaction cost: TRANSACTION_COST per SIDE (a switch = sell + buy = 2×)
  • Optional cross-sectional momentum ranking (config CROSS_SECTIONAL_RANK)
  • Switch requires the edge to persist SWITCH_CONFIRM_DAYS consecutive days
  • Benchmark: SPY buy-and-hold over the same period
  • Survivorship note: uses current index constituents → slight upward bias

Usage
─────
  python backtest.py                                  # 1 year, $10 000
  python backtest.py --start 2023-01-01               # custom start
  python backtest.py --start 2023-01-01 --end 2024-06-01 --capital 25000
  python backtest.py --fast                           # top-150 universe
  python backtest.py --start 2023-01-01 --no-chart    # skip PNG

Outputs
───────
  backtest_equity.csv   — daily portfolio value + SPY benchmark
  backtest_trades.csv   — every hold/switch decision + P&L per trade
  backtest_chart.png    — equity-curve chart (requires matplotlib)
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional, Union

# ── Make sure we can import sibling modules ───────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    BACKTEST_NORMALIZE_SCORE,
    CROSS_SECTIONAL_RANK,
    HIGH_WINDOW_DAYS,
    MIN_AVG_VOLUME,
    MIN_HOLD_DAYS,
    MIN_PRICE,
    REGIME_CONFIRM_DAYS,
    REGIME_FILTER_ENABLED,
    REGIME_REENTRY_COOLDOWN,
    REGIME_SMA_DAYS,
    STOP_LOSS_PCT,
    SWITCH_CONFIRM_DAYS,
    SWITCH_THRESHOLD,
    VOLUME_WINDOW_DAYS,
    WEIGHTS,
)
from signals import momentum_raw, rank_pct, regime_is_bullish, volume_score
from universe import get_universe

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence yfinance's misleading per-ticker "possibly delisted" errors —
# usable-ticker counts are summarized after each download instead.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ── Constants ─────────────────────────────────────────────────────────────────
TRANSACTION_COST = 0.001     # 0.1% per SIDE (sell and buy each pay this)
MIN_HISTORY_ROWS = HIGH_WINDOW_DAYS + VOLUME_WINDOW_DAYS + 5
EVENT_NEUTRAL    = 0.50      # placeholder for event signal (no historical data)
FLOW_NEUTRAL     = 0.50      # placeholder for flow signal  (no historical data)
DEFAULT_CAPITAL  = 10_000.0
DEFAULT_LOOKBACK_YEARS = 1
DEFAULT_MAX_UNIVERSE   = 300  # top N by avg volume; increase for more coverage


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the Daily Long-Only Rotation Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: 1 year ago)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: yesterday)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_CAPITAL,
        help=f"Starting capital in USD (default: {DEFAULT_CAPITAL:,.0f})",
    )
    parser.add_argument(
        "--max-universe",
        type=int,
        default=DEFAULT_MAX_UNIVERSE,
        help=(
            f"Max universe size, sorted by avg volume (default: {DEFAULT_MAX_UNIVERSE}). "
            "Increase for broader coverage, decrease for speed."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: limit universe to top 150 by volume",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip generating the PNG chart",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            f"Override SWITCH_THRESHOLD from config.py "
            f"(default: {SWITCH_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--include-event-flow",
        action="store_true",
        help=(
            "Include point-in-time Event/Flow historical proxies. Event uses "
            "directional price reaction + abnormal volume; Flow uses the last "
            "completed stock move + contemporaneous SPY trend."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data download
# ─────────────────────────────────────────────────────────────────────────────

def download_history(
    tickers: list[str],
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    """
    Bulk-download OHLCV for all tickers from (start - warm-up buffer) to end.
    Returns {ticker: DataFrame} for tickers with sufficient rows.
    """
    # Add warm-up buffer so signals have data from day 1 of the test range
    buffer_start = start - timedelta(days=MIN_HISTORY_ROWS * 2)

    logger.info(
        f"Downloading history for {len(tickers)} tickers "
        f"({buffer_start} → {end})…"
    )

    raw = yf.download(
        tickers,
        start=buffer_start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=True,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )

    result: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = raw[t].copy() if len(tickers) > 1 else raw.copy()
            df.dropna(how="all", inplace=True)
            if len(df) >= MIN_HISTORY_ROWS:
                result[t] = df
        except Exception:
            pass

    logger.info(f"Usable data: {len(result)} / {len(tickers)} tickers")
    return result


def download_spy(start: date, end: date) -> pd.Series:
    """SPY close prices for benchmark comparison."""
    buffer_start = start - timedelta(days=10)
    spy = yf.download(
        "SPY",
        start=buffer_start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True,
    )
    return spy["Close"].squeeze().dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Universe preparation
# ─────────────────────────────────────────────────────────────────────────────

def rank_by_liquidity(
    data: dict[str, pd.DataFrame],
    max_n: int,
) -> list[str]:
    """
    Rank tickers by average daily volume over available history.
    Returns top max_n tickers. Applies MIN_AVG_VOLUME and MIN_PRICE filters.
    """
    liquid = []
    for t, df in data.items():
        try:
            avg_vol   = float(df["Volume"].iloc[-VOLUME_WINDOW_DAYS:].mean())
            avg_price = float(df["Close"].iloc[-VOLUME_WINDOW_DAYS:].mean())
            if avg_vol >= MIN_AVG_VOLUME and avg_price >= MIN_PRICE:
                liquid.append((t, avg_vol))
        except Exception:
            pass

    liquid.sort(key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in liquid[:max_n]]
    logger.info(
        f"Liquidity filter: {len(liquid)} pass, using top {len(selected)} by volume"
    )
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Scoring (backtest-safe version)
# ─────────────────────────────────────────────────────────────────────────────

def combine_mv(m: float, v: float) -> float:
    """
    Combine momentum + volume components into the backtest composite.
    Event and flow are fixed at neutral (0.5) — historical data unavailable.

    When BACKTEST_NORMALIZE_SCORE=True (recommended), only the active
    signals are used, scaled to their combined weight → full [0, 1] range.
    """
    if BACKTEST_NORMALIZE_SCORE:
        active_w = WEIGHTS["momentum"] + WEIGHTS["volume"]
        return (WEIGHTS["momentum"] * m + WEIGHTS["volume"] * v) / active_w
    return (
        WEIGHTS["momentum"] * m +
        WEIGHTS["volume"]   * v +
        WEIGHTS["event"]    * EVENT_NEUTRAL +
        WEIGHTS["flow"]     * FLOW_NEUTRAL
    )


def event_proxy_score(hist: pd.DataFrame) -> float:
    """Point-in-time proxy for the price response to a company event.

    Historical Yahoo news/calendar snapshots are unavailable.  A directional
    close-to-close move confirmed by abnormal volume is used instead.  Only
    completed bars in ``hist`` are inspected, so this is backtest-safe.
    """
    if len(hist) < VOLUME_WINDOW_DAYS + 2:
        return EVENT_NEUTRAL
    close = hist["Close"].squeeze()
    volume = hist["Volume"].squeeze()
    prev_close = float(close.iloc[-2])
    avg_volume = float(volume.iloc[-(VOLUME_WINDOW_DAYS + 1):-1].mean())
    if prev_close <= 0 or avg_volume <= 0:
        return EVENT_NEUTRAL
    move = float(close.iloc[-1]) / prev_close - 1.0
    volume_ratio = max(float(volume.iloc[-1]) / avg_volume, 0.01)
    # Quiet sessions stay near neutral; large, volume-confirmed reactions
    # approach the ends of the [0, 1] range.
    impact = np.tanh(move * 18.0) * min(max(np.log(volume_ratio) + 1.0, 0.0), 2.0) / 2.0
    return float(np.clip(0.5 + 0.5 * impact, 0.0, 1.0))


def flow_proxy_score(hist: pd.DataFrame, spy_slice: Optional[pd.Series]) -> float:
    """End-of-day historical analogue of live latest-price + market flow.

    It uses the last completed stock return and the contemporaneous SPY
    five-session trend.  This deliberately avoids using the next session's
    open, which would make an at-open fill optimistic/look-ahead biased.
    """
    if len(hist) < 2:
        return FLOW_NEUTRAL
    close = hist["Close"].squeeze()
    prev_close = float(close.iloc[-2])
    if prev_close <= 0:
        return FLOW_NEUTRAL
    latest_move = float(close.iloc[-1]) / prev_close - 1.0
    score = 0.5 + (1.0 / (1.0 + np.exp(-40.0 * latest_move)) - 0.5)
    if spy_slice is not None and len(spy_slice.dropna()) >= 6:
        spy_close = spy_slice.dropna()
        spy_trend = float(spy_close.iloc[-1] / spy_close.iloc[-6] - 1.0)
        score += (1.0 / (1.0 + np.exp(-20.0 * spy_trend)) - 0.5) * 0.3
    return float(np.clip(score, 0.0, 1.0))


def combine_four(m: float, v: float, e: float, f: float) -> float:
    """Combine all four signals in the same raw units as the live scanner."""
    return (
        WEIGHTS["momentum"] * m + WEIGHTS["volume"] * v
        + WEIGHTS["event"] * e + WEIGHTS["flow"] * f
    )


def score_universe(
    universe_tickers: list,
    data: dict,
    prev_day,
    spy_slice: Optional[pd.Series],
    include_event_flow: bool = False,
) -> dict:
    """
    Score every ticker on data up to prev_day (no lookahead).

    With CROSS_SECTIONAL_RANK the momentum component is each stock's
    percentile rank of raw (unclipped) momentum across today's universe —
    removes the saturation ties at 1.0 that made picks liquidity-order
    dependent.
    """
    raws: dict = {}
    vols: dict = {}
    for t in universe_tickers:
        if t not in data:
            continue
        hist_slice = data[t].loc[:prev_day]
        if len(hist_slice) < MIN_HISTORY_ROWS:
            continue
        try:
            raws[t] = momentum_raw(hist_slice, spy_slice=spy_slice)
            vols[t] = volume_score(hist_slice)
        except Exception:
            pass

    if CROSS_SECTIONAL_RANK:
        ms = rank_pct(raws)
    else:
        ms = {t: min(max(r, 0.0), 1.0) for t, r in raws.items()}

    if not include_event_flow:
        return {t: combine_mv(ms[t], vols[t]) for t in ms}

    scores = {}
    for ticker in ms:
        if ticker not in vols:
            continue
        hist = data[ticker].loc[:prev_day]
        scores[ticker] = combine_four(
            ms[ticker], vols[ticker], event_proxy_score(hist),
            flow_proxy_score(hist, spy_slice),
        )
    return scores


def _px(df: pd.DataFrame, day, col: str) -> Optional[float]:
    """Price at (day, col), or None if missing/NaN/non-positive."""
    try:
        val = float(df.loc[day, col])
        return val if val > 0 and not np.isnan(val) else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    universe_tickers: list,
    data: dict,
    trading_days: pd.DatetimeIndex,
    initial_capital: float,
    switch_threshold: float,
    spy_series: Optional[pd.Series] = None,
    include_event_flow: bool = False,
) -> tuple:
    """
    Simulate the Hold/Switch strategy over the given trading days. (v3)

    v3 additions vs v2:
      - Per-trade stop loss (STOP_LOSS_PCT): exit if position drops below
        threshold from entry price, regardless of scoring
      - Regime confirmation (REGIME_CONFIRM_DAYS): SPY must be above SMA
        for N consecutive days before re-entering after a bear period
      - Regime cooldown (REGIME_REENTRY_COOLDOWN): mandatory cash days
        after any REGIME_EXIT before new entries are permitted
    """
    capital     = initial_capital
    position    = "CASH"
    equity_rows = []
    trade_rows  = []

    entry_price:            Optional[float] = None
    entry_date:             Optional[date]  = None
    hold_days:              int             = 0
    regime_exit_day_idx:    int             = -9999   # index of last regime exit
    consecutive_bull_days:  int             = 0       # days SPY has been above SMA
    edge_streak:            int             = 0       # consecutive days edge > threshold

    spy_close: Optional[pd.Series] = None
    if spy_series is not None:
        spy_close = spy_series.copy()
        spy_close.index = pd.to_datetime(spy_close.index)

    logger.info(
        f"Running simulation v4: {len(trading_days)} days, "
        f"{len(universe_tickers)} tickers, threshold={switch_threshold}, "
        f"stop_loss={STOP_LOSS_PCT}, regime_confirm={REGIME_CONFIRM_DAYS}, "
        f"cooldown={REGIME_REENTRY_COOLDOWN}, "
        f"switch_confirm={SWITCH_CONFIRM_DAYS}, "
        f"xsect_rank={CROSS_SECTIONAL_RANK}, event_flow_proxy={include_event_flow}, "
        f"open-execution, per-side costs"
    )

    for i in range(1, len(trading_days)):
        prev_day = trading_days[i - 1]
        curr_day = trading_days[i]
        switched = False

        # ── Regime check ──────────────────────────────────────────────────────
        in_bull_regime   = True
        days_since_exit  = i - regime_exit_day_idx

        if REGIME_FILTER_ENABLED and spy_close is not None:
            spy_slice_regime = spy_close.loc[:prev_day]
            today_bull       = regime_is_bullish(spy_slice_regime)

            if today_bull:
                consecutive_bull_days += 1
            else:
                consecutive_bull_days = 0

            # Require N consecutive bull days before allowing re-entry
            in_bull_regime = (
                today_bull and
                consecutive_bull_days >= REGIME_CONFIRM_DAYS
            )

        # ── Cooldown check ────────────────────────────────────────────────────
        in_cooldown = days_since_exit < REGIME_REENTRY_COOLDOWN

        # ── Force liquidate if bear regime (exit at today's open) ─────────────
        if not in_bull_regime and position != "CASH":
            factor = 1.0
            if position in data:
                close_prev = _px(data[position], prev_day, "Close")
                open_t     = _px(data[position], curr_day, "Open")
                exit_price = open_t or close_prev
                if close_prev and exit_price:
                    factor = exit_price / close_prev   # overnight move to the open
                if entry_price is not None and exit_price:
                    pnl_pct = (exit_price - entry_price) / entry_price
                    trade_rows.append({
                        "entry_date":  entry_date,
                        "exit_date":   curr_day.date(),
                        "ticker":      position,
                        "entry_price": round(entry_price, 4),
                        "exit_price":  round(exit_price, 4),
                        "pnl_pct":     round(pnl_pct * 100, 3),
                        "exit_reason": "REGIME_EXIT",
                        "best_ticker": "CASH",
                        "best_score":  0.0,
                        "score_delta": 0.0,
                    })
            factor               *= (1 - TRANSACTION_COST)   # sell side
            capital              *= factor
            position              = "CASH"
            entry_price           = None
            entry_date            = None
            hold_days             = 0
            switched              = True
            regime_exit_day_idx   = i
            edge_streak           = 0

            equity_rows.append({
                "date":            curr_day.date(),
                "portfolio_value": round(capital, 4),
                "daily_return":    round((factor - 1) * 100, 4),
                "position":        "CASH",
                "switched":        switched,
                "regime":          "BEAR",
            })
            continue

        # ── Stop loss check (trigger on prev close, exit at today's open) ─────
        stop_triggered = False
        if (
            STOP_LOSS_PCT is not None
            and position != "CASH"
            and entry_price is not None
            and position in data
        ):
            try:
                close_prev = _px(data[position], prev_day, "Close")
                if close_prev is not None:
                    ret_from_entry = (close_prev - entry_price) / entry_price
                    if ret_from_entry <= STOP_LOSS_PCT:
                        stop_triggered = True
                        open_t     = _px(data[position], curr_day, "Open")
                        exit_price = open_t or close_prev
                        factor     = exit_price / close_prev
                        pnl_pct    = (exit_price - entry_price) / entry_price
                        trade_rows.append({
                            "entry_date":  entry_date,
                            "exit_date":   curr_day.date(),
                            "ticker":      position,
                            "entry_price": round(entry_price, 4),
                            "exit_price":  round(exit_price, 4),
                            "pnl_pct":     round(pnl_pct * 100, 3),
                            "exit_reason": "STOP_LOSS",
                            "best_ticker": "",
                            "best_score":  0.0,
                            "score_delta": 0.0,
                        })
                        factor     *= (1 - TRANSACTION_COST)   # sell side
                        capital    *= factor
                        position    = "CASH"
                        entry_price = None
                        entry_date  = None
                        hold_days   = 0
                        switched    = True
                        edge_streak = 0
            except Exception:
                stop_triggered = False

        if stop_triggered:
            equity_rows.append({
                "date":            curr_day.date(),
                "portfolio_value": round(capital, 4),
                "daily_return":    round((factor - 1) * 100, 4),
                "position":        "CASH",
                "switched":        True,
                "regime":          "BULL",
            })
            continue

        # ── Don't enter if in cooldown or regime not confirmed ────────────────
        if position == "CASH" and (not in_bull_regime or in_cooldown):
            equity_rows.append({
                "date":            curr_day.date(),
                "portfolio_value": round(capital, 4),
                "daily_return":    0.0,
                "position":        "CASH",
                "switched":        False,
                "regime":          "BULL" if in_bull_regime else "BEAR",
            })
            continue

        # ── Prepare SPY slice for relative-strength scoring ───────────────────
        spy_score_slice: Optional[pd.Series] = None
        if spy_close is not None:
            spy_score_slice = spy_close.loc[:prev_day]

        # ── Score universe (data up to prev_day only — no lookahead) ─────────
        scores = score_universe(
            universe_tickers, data, prev_day, spy_score_slice,
            include_event_flow=include_event_flow,
        )

        if not scores:
            equity_rows.append({
                "date":            curr_day.date(),
                "portfolio_value": round(capital, 4),
                "daily_return":    0.0,
                "position":        position,
                "switched":        False,
                "regime":          "BULL",
            })
            hold_days += 1
            continue

        sorted_scores          = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_ticker, best_score = sorted_scores[0]
        current_score           = scores.get(position, 0.5)
        min_hold_met            = (hold_days >= MIN_HOLD_DAYS) or (position == "CASH")

        # ── Hold/Switch decision ──────────────────────────────────────────────
        # The edge must persist SWITCH_CONFIRM_DAYS consecutive days before a
        # stock→stock switch fires; entries from CASH are immediate.
        do_switch = False
        if position == "CASH":
            do_switch = True   # bull + not cooldown guaranteed at this point
        else:
            has_edge = (
                min_hold_met
                and best_ticker != position
                and (best_score - current_score) > (
                    switch_threshold * (WEIGHTS["momentum"] + WEIGHTS["volume"])
                    if include_event_flow and BACKTEST_NORMALIZE_SCORE
                    else switch_threshold
                )
            )
            edge_streak = edge_streak + 1 if has_edge else 0
            do_switch   = has_edge and edge_streak >= SWITCH_CONFIRM_DAYS

        # ── Execute at today's open, then earn open→close on the new leg ─────
        daily_factor = 1.0

        if do_switch:
            # Sell old position at today's open (earn the overnight move)
            if position != "CASH" and position in data:
                close_prev = _px(data[position], prev_day, "Close")
                open_t     = _px(data[position], curr_day, "Open")
                exit_price = open_t or close_prev
                if close_prev and exit_price:
                    daily_factor *= exit_price / close_prev
                if entry_price is not None and exit_price:
                    pnl_pct = (exit_price - entry_price) / entry_price
                    trade_rows.append({
                        "entry_date":  entry_date,
                        "exit_date":   curr_day.date(),
                        "ticker":      position,
                        "entry_price": round(entry_price, 4),
                        "exit_price":  round(exit_price, 4),
                        "pnl_pct":     round(pnl_pct * 100, 3),
                        "exit_reason": "SWITCH",
                        "best_ticker": best_ticker,
                        "best_score":  round(best_score, 4),
                        "score_delta": round(best_score - current_score, 4),
                    })
                daily_factor *= (1 - TRANSACTION_COST)   # sell side

            # Buy new position at today's open, earn open→close
            daily_factor *= (1 - TRANSACTION_COST)       # buy side
            new_open  = _px(data[best_ticker], curr_day, "Open")
            new_close = _px(data[best_ticker], curr_day, "Close")
            new_prev  = _px(data[best_ticker], prev_day, "Close")
            if new_open and new_close:
                entry_price   = new_open
                daily_factor *= new_close / new_open
            elif new_prev and new_close:
                entry_price   = new_prev     # fallback: no open print that day
                daily_factor *= new_close / new_prev
            else:
                entry_price = new_prev or new_close

            entry_date  = curr_day.date()
            position    = best_ticker
            hold_days   = 0
            switched    = True
            edge_streak = 0

        elif position != "CASH" and position in data:
            p0 = _px(data[position], prev_day, "Close")
            p1 = _px(data[position], curr_day, "Close")
            if p0 and p1:
                daily_factor = p1 / p0

        capital   *= daily_factor
        hold_days += 1

        equity_rows.append({
            "date":            curr_day.date(),
            "portfolio_value": round(capital, 4),
            "daily_return":    round((daily_factor - 1) * 100, 4),
            "position":        position,
            "switched":        switched,
            "regime":          "BULL",
        })

    equity_df = pd.DataFrame(equity_rows).set_index("date")
    trades_df = pd.DataFrame(trade_rows)
    return equity_df, trades_df





# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float,
    spy_series: pd.Series,
    start: date,
    end: date,
) -> dict:
    """Compute all performance metrics for display and export."""
    prices   = equity_df["portfolio_value"]
    daily_r  = equity_df["daily_return"] / 100.0

    # ── Returns ───────────────────────────────────────────────────────────────
    total_ret = (prices.iloc[-1] - initial_capital) / initial_capital
    n_days    = len(equity_df)
    ann_ret   = (1 + total_ret) ** (252 / max(n_days, 1)) - 1

    # ── SPY benchmark ─────────────────────────────────────────────────────────
    spy_aligned = spy_series.reindex(
        pd.to_datetime(equity_df.index), method="ffill"
    ).dropna()
    if len(spy_aligned) >= 2:
        spy_ret = float(spy_aligned.iloc[-1] / spy_aligned.iloc[0] - 1)
        spy_ann = (1 + spy_ret) ** (252 / max(len(spy_aligned), 1)) - 1
    else:
        spy_ret = spy_ann = 0.0

    # ── Volatility & Sharpe ───────────────────────────────────────────────────
    if len(daily_r) > 1:
        vol_ann   = float(daily_r.std() * np.sqrt(252))
        excess    = daily_r.mean() * 252 - 0.05   # vs 5% risk-free rate
        sharpe    = excess / vol_ann if vol_ann > 0 else 0.0
    else:
        vol_ann = sharpe = 0.0

    # ── Max drawdown ──────────────────────────────────────────────────────────
    running_max = prices.cummax()
    drawdowns   = (prices - running_max) / running_max
    max_dd      = float(drawdowns.min())

    # ── Calmar ────────────────────────────────────────────────────────────────
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else float("nan")

    # ── Win rate ──────────────────────────────────────────────────────────────
    active_days = daily_r[equity_df["position"] != "CASH"]
    win_rate    = float((active_days > 0).mean()) if len(active_days) > 0 else 0.0

    avg_win  = float(active_days[active_days > 0].mean() * 100) if (active_days > 0).any()  else 0.0
    avg_loss = float(active_days[active_days < 0].mean() * 100) if (active_days < 0).any() else 0.0

    # ── Profit factor ─────────────────────────────────────────────────────────
    gross_up   = active_days[active_days > 0].sum()
    gross_down = abs(active_days[active_days < 0].sum())
    pf = gross_up / gross_down if gross_down > 0 else float("nan")

    # ── Trades ────────────────────────────────────────────────────────────────
    n_switches   = int(equity_df["switched"].sum())
    turnover     = n_switches / max(n_days, 1)
    avg_hold_days = n_days / max(n_switches, 1)

    return {
        # Strategy
        "total_return_pct":  round(total_ret * 100, 2),
        "ann_return_pct":    round(ann_ret * 100, 2),
        "volatility_pct":    round(vol_ann * 100, 2),
        "sharpe":            round(sharpe, 3),
        "max_drawdown_pct":  round(max_dd * 100, 2),
        "calmar":            round(calmar, 3) if not np.isnan(calmar) else "n/a",
        # Activity
        "win_rate_pct":      round(win_rate * 100, 1),
        "avg_win_pct":       round(avg_win, 3),
        "avg_loss_pct":      round(avg_loss, 3),
        "profit_factor":     round(pf, 3) if not np.isnan(pf) else "n/a",
        "n_switches":        n_switches,
        "avg_hold_days":     round(avg_hold_days, 1),
        "turnover_daily":    round(turnover, 4),
        # Benchmark
        "spy_total_ret_pct": round(spy_ret * 100, 2),
        "spy_ann_ret_pct":   round(spy_ann * 100, 2),
        "alpha_ann_pct":     round((ann_ret - spy_ann) * 100, 2),
        # Meta
        "n_trading_days":    n_days,
        "start":             start.isoformat(),
        "end":               end.isoformat(),
        "initial_capital":   initial_capital,
        "final_value":       round(prices.iloc[-1], 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def _divider(char="═", width=62):
    print(char * width)


def print_summary(m: dict, switch_threshold: float, include_event_flow: bool = False):
    print()
    _divider()
    print("  BACKTEST RESULTS — Hold/Switch Rotation Strategy")
    print(f"  Period  : {m['start']}  →  {m['end']}  ({m['n_trading_days']} trading days)")
    print(f"  Capital : ${m['initial_capital']:>10,.2f}  →  ${m['final_value']:>10,.2f}")
    print(f"  Threshold used : {switch_threshold}")
    _divider("─")
    print("  RETURNS")
    print(f"    Strategy total          : {m['total_return_pct']:>+8.2f}%")
    print(f"    Strategy annualised     : {m['ann_return_pct']:>+8.2f}%")
    print(f"    SPY total               : {m['spy_total_ret_pct']:>+8.2f}%")
    print(f"    SPY annualised          : {m['spy_ann_ret_pct']:>+8.2f}%")
    print(f"    Alpha (ann.)            : {m['alpha_ann_pct']:>+8.2f}%")
    _divider("─")
    print("  RISK")
    print(f"    Annualised volatility   : {m['volatility_pct']:>8.2f}%")
    print(f"    Max drawdown            : {m['max_drawdown_pct']:>8.2f}%")
    print(f"    Sharpe ratio            : {m['sharpe']:>8.3f}")
    print(f"    Calmar ratio            : {m['calmar']!s:>8}")
    _divider("─")
    print("  ACTIVITY")
    print(f"    Win rate                : {m['win_rate_pct']:>8.1f}%")
    print(f"    Avg win  / Avg loss     : {m['avg_win_pct']:>+6.3f}% / {m['avg_loss_pct']:>+6.3f}%")
    print(f"    Profit factor           : {m['profit_factor']!s:>8}")
    print(f"    Number of switches      : {m['n_switches']:>8}")
    print(f"    Avg hold (days)         : {m['avg_hold_days']:>8.1f}")
    print(f"    Daily turnover          : {m['turnover_daily']:>8.4f}")
    _divider("─")
    if include_event_flow:
        print("  NOTE: Backtest uses Momentum + Volume + Event/Flow proxies.")
        print("  Proxies use point-in-time OHLCV/market data, not historical news snapshots.")
    else:
        print("  NOTE: Backtest uses Momentum + Volume signals only.")
        print("  Event & Flow signals (unavailable historically) set to neutral.")
    print("  Trades execute at next-day OPEN; costs charged per side.")
    print("  Survivorship bias present (current index constituents used).")
    _divider()
    print()


def save_equity_csv(
    equity_df: pd.DataFrame,
    spy_series: pd.Series,
    initial_capital: float,
    path: str = "backtest_equity.csv",
):
    df = equity_df.copy()
    spy_aligned = spy_series.reindex(
        pd.to_datetime(df.index), method="ffill"
    )
    spy_base = spy_aligned.iloc[0] if len(spy_aligned) > 0 else 1.0
    df["spy_value"] = (spy_aligned / spy_base * initial_capital).values
    df.to_csv(path)
    logger.info(f"Equity curve saved → {path}")


def save_trades_csv(trades_df: pd.DataFrame, path: str = "backtest_trades.csv"):
    if trades_df.empty:
        logger.info("No completed trades to save")
        return
    trades_df.to_csv(path, index=False)
    logger.info(f"Trade log saved → {path}")


def generate_chart(
    equity_df: pd.DataFrame,
    spy_series: pd.Series,
    initial_capital: float,
    metrics: dict,
    path: str = "backtest_chart.png",
):
    """Generate a 3-panel chart: equity curve, drawdown, daily returns."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.gridspec import GridSpec
    except ImportError:
        logger.warning("matplotlib not installed — skipping chart (pip install matplotlib)")
        return

    dates = pd.to_datetime(equity_df.index)

    # SPY normalised to same starting capital
    spy_aligned = spy_series.reindex(dates, method="ffill")
    spy_norm = spy_aligned / spy_aligned.iloc[0] * initial_capital

    # Portfolio
    port = equity_df["portfolio_value"]

    # Drawdown
    running_max = port.cummax()
    drawdown    = (port - running_max) / running_max * 100

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#0f1117")
    gs = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1], hspace=0.08)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        ax.spines[:].set_color("#333333")
        ax.yaxis.label.set_color("#aaaaaa")

    # ── Panel 1: Equity curve ─────────────────────────────────────────────────
    ax1.plot(dates, port,     color="#00e5ff", linewidth=1.5, label="Strategy", zorder=3)
    ax1.plot(dates, spy_norm, color="#ff6d00", linewidth=1.2, label="SPY",      zorder=2,
             alpha=0.85, linestyle="--")
    ax1.fill_between(dates, initial_capital, port,
                     where=(port >= initial_capital),
                     alpha=0.10, color="#00e5ff", interpolate=True)
    ax1.fill_between(dates, initial_capital, port,
                     where=(port <  initial_capital),
                     alpha=0.15, color="#ff4444", interpolate=True)
    ax1.axhline(initial_capital, color="#555555", linewidth=0.7, linestyle=":")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left", facecolor="#1a1a2e", edgecolor="#333333",
               labelcolor="white", fontsize=9)

    # Annotate final values
    ax1.annotate(
        f"${port.iloc[-1]:,.0f}  ({metrics['total_return_pct']:+.1f}%)",
        xy=(dates[-1], port.iloc[-1]),
        xytext=(-90, 10), textcoords="offset points",
        color="#00e5ff", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=0.8),
    )

    title = (
        f"Hold/Switch Rotation Strategy  ·  "
        f"{metrics['start']} → {metrics['end']}  ·  "
        f"Sharpe {metrics['sharpe']:.2f}  ·  "
        f"MaxDD {metrics['max_drawdown_pct']:.1f}%  ·  "
        f"{metrics['n_switches']} switches"
    )
    ax1.set_title(title, color="white", fontsize=10, pad=10)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────────
    ax2.fill_between(dates, drawdown, 0, alpha=0.6, color="#ff4444")
    ax2.plot(dates, drawdown, color="#ff4444", linewidth=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.axhline(0, color="#555555", linewidth=0.5)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── Panel 3: Daily returns ────────────────────────────────────────────────
    daily_r = equity_df["daily_return"]
    colors  = ["#00c853" if r >= 0 else "#ff4444" for r in daily_r]
    ax3.bar(dates, daily_r, color=colors, width=1.0, alpha=0.7)
    ax3.axhline(0, color="#555555", linewidth=0.5)
    ax3.set_ylabel("Daily Ret (%)")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right", color="#aaaaaa")

    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Chart saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Date range ────────────────────────────────────────────────────────────
    end   = date.fromisoformat(args.end)   if args.end   else date.today() - timedelta(days=1)
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=365)

    if start >= end:
        logger.error("--start must be before --end")
        sys.exit(1)

    # ── Threshold ─────────────────────────────────────────────────────────────
    switch_threshold = args.threshold if args.threshold is not None else SWITCH_THRESHOLD

    # ── Universe size ─────────────────────────────────────────────────────────
    max_uni = 150 if args.fast else args.max_universe

    print()
    print("════════════════════════════════════════════════════════════════")
    print("  HOLD/SWITCH ROTATION STRATEGY — BACKTEST")
    print(f"  Period  : {start}  →  {end}")
    print(f"  Capital : ${args.capital:,.2f}    Universe cap: {max_uni}    Threshold: {switch_threshold}")
    print("════════════════════════════════════════════════════════════════")

    # ── Step 1: Universe ──────────────────────────────────────────────────────
    print("\n  [1/5]  Fetching universe…")
    raw_universe = get_universe()
    if not raw_universe:
        logger.error("Empty universe — check internet connection.")
        sys.exit(1)
    print(f"         {len(raw_universe)} tickers from configured sources")

    # ── Step 2: Download OHLCV ────────────────────────────────────────────────
    print("\n  [2/5]  Downloading historical OHLCV…")
    all_data = download_history(raw_universe, start, end)

    # ── Step 3: Liquidity filter → top N by volume ───────────────────────────
    print("\n  [3/5]  Applying liquidity filter…")
    universe = rank_by_liquidity(all_data, max_uni)
    # Reduce data dict to selected universe (saves memory)
    data = {t: all_data[t] for t in universe if t in all_data}
    print(f"         Using {len(data)} tickers for simulation")

    # ── Step 4: SPY benchmark ─────────────────────────────────────────────────
    print("\n  [4/5]  Downloading SPY benchmark…")
    spy_series = download_spy(start, end)

    # ── Step 5: Get trading days ──────────────────────────────────────────────
    # Use the union of all dates present in the data
    all_dates: set = set()
    for df in data.values():
        all_dates.update(df.index)

    trading_days = pd.DatetimeIndex(
        sorted(d for d in all_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end))
    )

    if len(trading_days) < 10:
        logger.error(f"Only {len(trading_days)} trading days in range — too short to backtest")
        sys.exit(1)

    print(f"         {len(trading_days)} trading days in simulation range")

    # ── Step 6: Simulate ──────────────────────────────────────────────────────
    regime_label = f"ON (SPY > {REGIME_SMA_DAYS}-day SMA)" if REGIME_FILTER_ENABLED else "OFF"
    print(f"\n  [5/5]  Running simulation…  (regime filter: {regime_label})")
    equity_df, trades_df = run_simulation(
        universe_tickers=universe,
        data=data,
        trading_days=trading_days,
        initial_capital=args.capital,
        switch_threshold=switch_threshold,
        spy_series=spy_series,
        include_event_flow=args.include_event_flow,
    )

    # ── Regime stats ──────────────────────────────────────────────────────────
    if "regime" in equity_df.columns:
        bear_days = int((equity_df["regime"] == "BEAR").sum())
        bull_days = int((equity_df["regime"] == "BULL").sum())
        print(f"         Bull days: {bull_days}  |  Bear (cash) days: {bear_days}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(
        equity_df, trades_df, args.capital, spy_series, start, end
    )

    # ── Output ────────────────────────────────────────────────────────────────
    print_summary(metrics, switch_threshold, args.include_event_flow)

    save_equity_csv(equity_df, spy_series, args.capital)
    save_trades_csv(trades_df)

    if not args.no_chart:
        generate_chart(equity_df, spy_series, args.capital, metrics)
    else:
        print("  Chart skipped (--no-chart)")

    print(f"  Files written:")
    print(f"    backtest_equity.csv   — daily portfolio value + SPY")
    print(f"    backtest_trades.csv   — completed trade log")
    if not args.no_chart:
        print(f"    backtest_chart.png    — equity curve chart")
    print()


if __name__ == "__main__":
    main()
