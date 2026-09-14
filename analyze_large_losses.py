"""Find entry-time characteristics shared by large legacy-strategy losses."""

from __future__ import annotations

import argparse
from datetime import date

import numpy as np
import pandas as pd

import analyze_old_momentum_event as strategy


bt = strategy.legacy


def entry_features(frame: pd.DataFrame, entry_day) -> dict:
    hist = frame.loc[:pd.Timestamp(entry_day)].copy()
    close = hist["Close"].astype(float)
    high = hist["High"].astype(float)
    low = hist["Low"].astype(float)
    volume = hist["Volume"].astype(float)
    returns = close.pct_change()
    prior_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prior_close).abs(), (low - prior_close).abs()],
        axis=1,
    ).max(axis=1)
    ma20 = close.iloc[-20:].mean()
    high20 = high.iloc[-20:].max()
    avg_volume20 = volume.iloc[-21:-1].mean()
    return {
        "ret_1d": close.iloc[-1] / close.iloc[-2] - 1,
        "ret_5d": close.iloc[-1] / close.iloc[-6] - 1,
        "ret_20d": close.iloc[-1] / close.iloc[-21] - 1,
        "daily_vol_10d": returns.iloc[-10:].std(),
        "daily_vol_20d": returns.iloc[-20:].std(),
        "atr_14_pct": true_range.iloc[-14:].mean() / close.iloc[-1],
        "volume_ratio_20d": volume.iloc[-1] / avg_volume20,
        "above_ma20_pct": close.iloc[-1] / ma20 - 1,
        "from_high20_pct": close.iloc[-1] / high20 - 1,
        "event_proxy": strategy.event_proxy(hist),
        "entry_price": close.iloc[-1],
    }


def parabolic_or_low_price_surge(hist: pd.DataFrame) -> bool:
    """Candidate screen derived only from information available at entry."""
    close = hist["Close"].astype(float)
    if len(close) < 21:
        return False
    returns = close.pct_change()
    ret_1d = close.iloc[-1] / close.iloc[-2] - 1
    ret_20d = close.iloc[-1] / close.iloc[-21] - 1
    daily_vol_10d = returns.iloc[-10:].std()
    return bool(
        (ret_20d > .30 and daily_vol_10d > .04)
        or (close.iloc[-1] < 15 and ret_1d > .10)
    )


def intraday_stop_outcome(frame: pd.DataFrame, trade: dict) -> dict:
    """Daily-OHLC approximation of a broker stop active after entry close.

    If a session opens through the stop, the assumed fill is that day's open;
    otherwise it is the stop price. This cannot model sub-minute slippage.
    """
    entry_price = float(trade["entry_price"])
    stop_price = entry_price * 0.94
    entry_day = pd.Timestamp(trade["entry_date"])
    exit_day = pd.Timestamp(trade["exit_date"])
    holding = frame.loc[(frame.index > entry_day) & (frame.index <= exit_day)]
    for day, bar in holding.iterrows():
        day_low = float(bar["Low"])
        if day_low > stop_price:
            continue
        day_open = float(bar["Open"])
        fill = day_open if day_open <= stop_price else stop_price
        return {
            "intraday_stop_triggered": True,
            "intraday_stop_date": day.date(),
            "intraday_stop_fill": fill,
            "intraday_stop_pnl_pct": (fill / entry_price - 1) * 100,
        }
    return {
        "intraday_stop_triggered": False,
        "intraday_stop_date": None,
        "intraday_stop_fill": np.nan,
        "intraday_stop_pnl_pct": np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-09-14")
    parser.add_argument("--end", default="2026-09-11")
    parser.add_argument("--max-universe", type=int, default=150)
    args = parser.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    raw_universe = bt.get_universe()
    all_data = bt.download_history(raw_universe, start, end)
    universe = bt.rank_by_liquidity(all_data, args.max_universe)
    data = {ticker: all_data[ticker] for ticker in universe}
    spy = bt.download_spy(start, end)
    days = pd.DatetimeIndex(sorted({
        d for frame in data.values() for d in frame.index
        if pd.Timestamp(start) <= d <= pd.Timestamp(end)
    }))
    equity, trades = bt.run_simulation(universe, data, days, 10_000, 0.18, spy)

    rows = []
    for trade in trades.to_dict("records"):
        ticker = trade["ticker"]
        try:
            row = dict(trade)
            row.update(entry_features(data[ticker], trade["entry_date"]))
            row.update(intraday_stop_outcome(data[ticker], trade))
            rows.append(row)
        except Exception:
            continue
    sample = pd.DataFrame(rows)
    sample["large_loss"] = sample["pnl_pct"] <= -10
    feature_cols = [
        "ret_1d", "ret_5d", "ret_20d", "daily_vol_10d",
        "daily_vol_20d", "atr_14_pct", "volume_ratio_20d",
        "above_ma20_pct", "from_high20_pct", "event_proxy", "entry_price",
    ]

    print("\nSUMMARY")
    print({
        "period": f"{start} to {end}", "trades": len(sample),
        "large_losses": int(sample.large_loss.sum()),
        "final_value": round(float(equity.portfolio_value.iloc[-1]), 2),
    })
    print("\nLARGE LOSSES")
    show = ["entry_date", "exit_date", "ticker", "pnl_pct", "exit_reason"] + feature_cols
    print(sample.loc[sample.large_loss, show].to_string(index=False))

    triggered = sample[sample.intraday_stop_triggered]
    caught_large = sample[sample.large_loss & sample.intraday_stop_triggered]
    print("\nINTRADAY 6% STOP APPROXIMATION (daily OHLC)")
    print({
        "all_trades_triggered": len(triggered),
        "all_trades_triggered_pct": round(len(triggered) / len(sample) * 100, 2),
        "large_losses_caught": len(caught_large),
        "large_losses_total": int(sample.large_loss.sum()),
        "median_assumed_stop_pnl_pct": round(
            float(triggered.intraday_stop_pnl_pct.median()), 3
        ) if len(triggered) else None,
        "worst_assumed_stop_pnl_pct": round(
            float(triggered.intraday_stop_pnl_pct.min()), 3
        ) if len(triggered) else None,
    })
    if len(caught_large):
        print(caught_large[[
            "ticker", "entry_date", "exit_date", "pnl_pct",
            "intraday_stop_date", "intraday_stop_pnl_pct",
        ]].to_string(index=False))

    print("\nGROUP MEDIANS (large loss vs other trades)")
    comparison = pd.DataFrame({
        "large_loss": sample.loc[sample.large_loss, feature_cols].median(),
        "other": sample.loc[~sample.large_loss, feature_cols].median(),
    })
    comparison["difference"] = comparison.large_loss - comparison.other
    print(comparison.to_string())

    rules = []
    candidates = {
        "1d return > 3%": sample.ret_1d > .03,
        "1d return > 5%": sample.ret_1d > .05,
        "1d return > 8%": sample.ret_1d > .08,
        "5d return > 10%": sample.ret_5d > .10,
        "5d return > 15%": sample.ret_5d > .15,
        "20d return > 20%": sample.ret_20d > .20,
        "20d return > 30%": sample.ret_20d > .30,
        "10d daily vol > 3%": sample.daily_vol_10d > .03,
        "10d daily vol > 4%": sample.daily_vol_10d > .04,
        "ATR14 > 4%": sample.atr_14_pct > .04,
        "ATR14 > 5%": sample.atr_14_pct > .05,
        "volume ratio > 1.5x": sample.volume_ratio_20d > 1.5,
        "volume ratio > 2x": sample.volume_ratio_20d > 2,
        "above MA20 > 8%": sample.above_ma20_pct > .08,
        "above MA20 > 12%": sample.above_ma20_pct > .12,
        "Event proxy > 0.65": sample.event_proxy > .65,
        "Event proxy > 0.75": sample.event_proxy > .75,
        "price < $15": sample.entry_price < 15,
        "20d >30% AND vol10 >4%": (
            (sample.ret_20d > .30) & (sample.daily_vol_10d > .04)
        ),
        "20d >30% AND above MA20 >12%": (
            (sample.ret_20d > .30) & (sample.above_ma20_pct > .12)
        ),
        "vol10 >4% AND above MA20 >12%": (
            (sample.daily_vol_10d > .04) & (sample.above_ma20_pct > .12)
        ),
        "parabolic OR low-price surge": (
            (
                (sample.ret_20d > .30)
                & (sample.daily_vol_10d > .04)
            )
            | (
                (sample.entry_price < 15)
                & (sample.ret_1d > .10)
            )
        ),
    }
    positives = sample.large_loss
    for name, mask in candidates.items():
        recall = mask[positives].mean() if positives.any() else np.nan
        false_exclusion = mask[~positives].mean() if (~positives).any() else np.nan
        rules.append({
            "rule": name,
            "large_loss_catch_pct": recall * 100,
            "other_trades_excluded_pct": false_exclusion * 100,
            "precision_pct": positives[mask].mean() * 100 if mask.any() else 0,
            "net_separation": recall - false_exclusion,
        })
    rule_frame = pd.DataFrame(rules).sort_values(
        ["net_separation", "large_loss_catch_pct"], ascending=False
    )
    print("\nENTRY-TIME EXCLUSION RULE SCREEN")
    print(rule_frame.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    base_metrics = bt.compute_metrics(equity, trades, 10_000, spy, start, end)
    original_score = bt.backtest_composite_score

    def filtered_score(hist, spy_slice=None):
        if parabolic_or_low_price_surge(hist):
            return -1.0
        return original_score(hist, spy_slice=spy_slice)

    bt.backtest_composite_score = filtered_score
    try:
        filtered_equity, filtered_trades = bt.run_simulation(
            universe, data, days, 10_000, 0.18, spy
        )
    finally:
        bt.backtest_composite_score = original_score
    filtered_metrics = bt.compute_metrics(
        filtered_equity, filtered_trades, 10_000, spy, start, end
    )
    print("\nFULL BACKTEST IMPACT OF BEST SCREEN")
    for label, metric in (("base", base_metrics), ("filtered", filtered_metrics)):
        print(label, {
            key: metric[key] for key in (
                "total_return_pct", "ann_return_pct", "sharpe",
                "max_drawdown_pct", "calmar", "n_switches", "final_value",
            )
        }, "large_losses", int((
            (trades if label == "base" else filtered_trades).pnl_pct <= -10
        ).sum()))


if __name__ == "__main__":
    main()
