"""Measure how much a 15:30 ET scan differs from daily-close proxy signals."""

from __future__ import annotations

import pickle
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import analyze_strategy_variants as analysis


CACHE = Path("/private/tmp/relay_strategy_variants_cache.pkl")
START = time(9, 30)
CUTOFF = time(15, 30)


def ticker_frame(raw: pd.DataFrame, ticker: str, multiple: bool) -> pd.DataFrame:
    try:
        frame = raw[ticker].copy() if multiple else raw.copy()
    except Exception:
        return pd.DataFrame()
    frame = frame.dropna(how="all")
    if frame.empty:
        return frame
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize("America/New_York")
    else:
        index = index.tz_convert("America/New_York")
    frame.index = index
    return frame


def partial_sessions(frame: pd.DataFrame, max_day: pd.Timestamp) -> dict[pd.Timestamp, dict]:
    result = {}
    for session_date, group in frame.groupby(frame.index.date):
        day = pd.Timestamp(session_date)
        if day > max_day:
            continue
        regular = group[
            (group.index.time >= START) & (group.index.time < CUTOFF)
        ]
        if regular.empty:
            continue
        result[day] = {
            "Open": float(regular["Open"].iloc[0]),
            "High": float(regular["High"].max()),
            "Low": float(regular["Low"].min()),
            "Close": float(regular["Close"].iloc[-1]),
            "Volume": float(regular["Volume"].sum()),
        }
    return result


def percentile_rank(values: dict[str, float]) -> pd.Series:
    series = pd.Series(values)
    if len(series) <= 1:
        return pd.Series(0.5, index=series.index)
    return (series.rank(method="average") - 1) / (len(series) - 1)


def main() -> None:
    if not CACHE.exists():
        raise SystemExit("Run analyze_strategy_variants.py first to create its cache.")
    with CACHE.open("rb") as handle:
        cached = pickle.load(handle)
    universe = cached["universe"]
    daily = cached["data"]
    close_scores = cached["scores"]["ranked_live"]
    max_day = pd.Timestamp(cached["days"].max())

    raw = yf.download(
        universe,
        period="60d",
        interval="5m",
        auto_adjust=True,
        group_by="ticker",
        prepost=False,
        threads=True,
        progress=False,
    )
    partial = {}
    drift_rows = []
    for ticker in universe:
        frame = ticker_frame(raw, ticker, len(universe) > 1)
        sessions = partial_sessions(frame, max_day)
        partial[ticker] = sessions
        for day, bar in sessions.items():
            try:
                official = float(daily[ticker].loc[day, "Close"])
            except Exception:
                continue
            drift_rows.append({
                "date": day,
                "ticker": ticker,
                "drift": official / bar["Close"] - 1,
            })

    available_days = sorted({day for sessions in partial.values() for day in sessions})
    comparison_rows = []
    for day in available_days:
        raw_momentum = {}
        events = {}
        prices_1530 = {}
        for ticker in universe:
            bar = partial[ticker].get(day)
            if bar is None:
                continue
            prior = daily[ticker].loc[daily[ticker].index < day]
            if len(prior) < analysis.bt.MIN_HISTORY_ROWS:
                continue
            current = pd.DataFrame([bar], index=[day])
            hist = pd.concat([prior, current])
            try:
                raw_momentum[ticker] = analysis._momentum_raw(hist, None)
                events[ticker] = analysis.strategy.event_proxy(hist)
                prices_1530[ticker] = bar["Close"]
            except Exception:
                continue
        if len(raw_momentum) < 30 or day not in close_scores.index:
            continue
        ranks = percentile_rank(raw_momentum)
        score_1530 = pd.Series({
            ticker: analysis.MOMENTUM_WEIGHT * float(ranks[ticker])
            + analysis.EVENT_WEIGHT * events[ticker]
            for ticker in ranks.index
        }).sort_values(ascending=False)
        score_close = close_scores.loc[day].dropna()
        common = score_1530.index.intersection(score_close.index)
        if len(common) < 30:
            continue
        top_1530 = score_1530.index[0]
        top_close = score_close.sort_values(ascending=False).index[0]
        close_order = score_close.loc[common].rank(ascending=False, method="min")
        try:
            official = float(daily[top_1530].loc[day, "Close"])
            top_drift = official / prices_1530[top_1530] - 1
        except Exception:
            top_drift = np.nan
        comparison_rows.append({
            "date": day,
            "coverage": len(common),
            "top_1530": top_1530,
            "top_close": top_close,
            "same_top": top_1530 == top_close,
            "top_1530_close_rank": float(close_order.get(top_1530, np.nan)),
            "rank_correlation": float(
                score_1530.loc[common].corr(score_close.loc[common], method="spearman")
            ),
            "top_drift": top_drift,
        })

    drift = pd.DataFrame(drift_rows)
    comparison = pd.DataFrame(comparison_rows)
    if drift.empty or comparison.empty:
        raise SystemExit("Not enough intraday data returned for comparison.")

    absolute_drift = drift["drift"].abs()
    print(f"Sessions compared: {len(comparison)}")
    print(f"Ticker-session observations: {len(drift)}")
    print(f"Median 15:30-to-close drift: {drift['drift'].median() * 100:.3f}%")
    print(f"Median absolute drift: {absolute_drift.median() * 100:.3f}%")
    print(f"90th percentile absolute drift: {absolute_drift.quantile(.90) * 100:.3f}%")
    print(f"99th percentile absolute drift: {absolute_drift.quantile(.99) * 100:.3f}%")
    print(f"Same top pick: {comparison['same_top'].mean() * 100:.1f}%")
    print(
        "15:30 top remains close top-3: "
        f"{(comparison['top_1530_close_rank'] <= 3).mean() * 100:.1f}%"
    )
    print(f"Median score rank correlation: {comparison['rank_correlation'].median():.3f}")
    print(f"Median top-pick 15:30-to-close return: {comparison['top_drift'].median() * 100:.3f}%")
    print(f"Mean top-pick 15:30-to-close return: {comparison['top_drift'].mean() * 100:.3f}%")
    print("\nLowest-agreement sessions")
    print(
        comparison.sort_values(["same_top", "rank_correlation"])
        .head(10)
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
