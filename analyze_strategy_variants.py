"""Compare risk and switching variants for the pre-close Relay strategy.

The long-history simulation uses daily OHLC. A decision is formed with data
through day t, executed at that close, and earns the close(t)->close(t+1)
return. Intraday stops use the following session's Open/Low and therefore do
not assume that a stop always fills exactly at its trigger after a gap.

Historical Event is necessarily the same point-in-time price/volume proxy used
by analyze_old_momentum_event.py; Yahoo does not provide historical news
snapshots. Results are an experiment, not a claim that live Event is identical.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from datetime import date
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_old_momentum_event as strategy


bt = strategy.legacy
signals = strategy.legacy_signals
MOMENTUM_WEIGHT = strategy.MOMENTUM_WEIGHT
EVENT_WEIGHT = strategy.EVENT_WEIGHT
CACHE_VERSION = 2


@dataclass(frozen=True)
class Variant:
    name: str
    score_mode: str = "absolute"
    regime: bool = True
    regime_sma: int = 50
    min_hold: int = 2
    threshold: float = 0.18
    fast_threshold: float | None = None
    positions: int = 1
    inverse_vol: bool = False
    target_vol: float | None = None
    stop_mode: str = "close"
    stop_pct: float = 0.06
    atr_multiple: float = 1.5


@dataclass
class Holding:
    entry_price: float
    entry_date: pd.Timestamp
    held_days: int
    stop_price: float | None
    weight: float


def _px(frame: pd.DataFrame, day: pd.Timestamp, column: str) -> float | None:
    try:
        value = float(frame.loc[day, column])
        return value if np.isfinite(value) and value > 0 else None
    except Exception:
        return None


def _daily_vol(frame: pd.DataFrame, day: pd.Timestamp, window: int = 20) -> float:
    close = frame.loc[:day, "Close"].astype(float)
    value = float(close.pct_change().iloc[-window:].std())
    return value if np.isfinite(value) and value > 0 else 0.02


def _atr_pct(frame: pd.DataFrame, day: pd.Timestamp, window: int = 14) -> float:
    hist = frame.loc[:day].tail(window + 1)
    close = hist["Close"].astype(float)
    prior = close.shift(1)
    true_range = pd.concat(
        [
            hist["High"].astype(float) - hist["Low"].astype(float),
            (hist["High"].astype(float) - prior).abs(),
            (hist["Low"].astype(float) - prior).abs(),
        ],
        axis=1,
    ).max(axis=1)
    last_close = float(close.iloc[-1])
    value = float(true_range.tail(window).mean() / last_close)
    return value if np.isfinite(value) and value > 0 else 0.04


def _momentum_raw(hist: pd.DataFrame, spy_slice: pd.Series | None) -> float:
    """Current scanner momentum formula before its final [0, 1] clipping."""
    min_rows = signals.HIGH_WINDOW_DAYS + signals.MOMENTUM_MEDIUM_DAYS + 2
    if len(hist) < min_rows:
        return 0.5
    close = hist["Close"].squeeze()
    high = hist["High"].squeeze()
    r1 = float(close.iloc[-1] / close.iloc[-2] - 1)
    r5 = float(close.iloc[-1] / close.iloc[-(signals.MOMENTUM_MEDIUM_DAYS + 1)] - 1)
    recent_vol = float(
        close.pct_change().iloc[-signals.VOL_PENALTY_DAYS:].dropna().std()
    )
    if signals.RISK_ADJUSTED_MOMENTUM and recent_vol > 0:
        r1_adj = r1 / recent_vol
        r5_adj = r5 / (recent_vol * sqrt(signals.MOMENTUM_MEDIUM_DAYS))
    else:
        r1_adj = r1 * 50
        r5_adj = r5 * 20

    rs_adj = 0.0
    if (
        signals.RELATIVE_STRENGTH_ENABLED
        and spy_slice is not None
        and len(spy_slice) >= signals.MOMENTUM_MEDIUM_DAYS + 1
    ):
        spy_r1 = float(spy_slice.iloc[-1] / spy_slice.iloc[-2] - 1)
        spy_r5 = float(
            spy_slice.iloc[-1]
            / spy_slice.iloc[-(signals.MOMENTUM_MEDIUM_DAYS + 1)]
            - 1
        )
        spy_vol = float(
            spy_slice.pct_change().iloc[-signals.VOL_PENALTY_DAYS:].dropna().std()
        )
        if spy_vol > 0:
            rel_r1 = (r1 - spy_r1) / spy_vol
            rel_r5 = r5 - spy_r5
            rel_r5 /= spy_vol * sqrt(signals.MOMENTUM_MEDIUM_DAYS)
            rs_adj = (
                (signals._sigmoid(rel_r1, k=1.5) - 0.5) * 0.20
                + (signals._sigmoid(rel_r5, k=1.0) - 0.5) * 0.10
            )

    high_20d = float(high.iloc[-signals.HIGH_WINDOW_DAYS:].max())
    pct_from_20h = float(close.iloc[-1]) / high_20d - 1
    prior_high = float(high.iloc[-(signals.HIGH_WINDOW_DAYS + 1):-1].max())
    breakout_bonus = 0.15 if float(close.iloc[-1]) > prior_high else 0.0
    ma10 = float(close.iloc[-10:].mean())
    trend_bonus = 0.05 if float(close.iloc[-1]) > ma10 else -0.05
    vol_penalty = 0.0
    if signals.VOL_PENALTY_ENABLED and recent_vol > signals.VOL_PENALTY_THRESHOLD:
        excess = recent_vol - signals.VOL_PENALTY_THRESHOLD
        vol_penalty = min(excess * signals.VOL_PENALTY_STRENGTH, 0.25)
    raw = (
        (
            signals._sigmoid(r1_adj, k=1.5)
            + signals._sigmoid(r5_adj, k=1.0)
            + signals._clip01(1.0 + pct_from_20h * 5.0)
        )
        / 3.0
        + breakout_bonus
        + trend_bonus
        + rs_adj
        - vol_penalty
    )
    return 0.5 if np.isnan(raw) else float(raw)


def build_score_matrices(
    universe: list[str],
    data: dict[str, pd.DataFrame],
    days: pd.DatetimeIndex,
    spy: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Precompute legacy absolute and cross-sectional Momentum+Event scores."""
    absolute_rows: list[dict] = []
    ranked_rows: list[dict] = []
    ranked_live_rows: list[dict] = []
    spy = spy.copy()
    spy.index = pd.to_datetime(spy.index)

    for number, day in enumerate(days, 1):
        spy_slice = spy.loc[:day]
        absolute: dict[str, float] = {}
        raw_momentum: dict[str, float] = {}
        raw_momentum_no_spy: dict[str, float] = {}
        event_values: dict[str, float] = {}
        for ticker in universe:
            hist = data[ticker].loc[:day]
            if len(hist) < bt.MIN_HISTORY_ROWS:
                continue
            try:
                momentum = signals.momentum_score(hist, spy_slice=spy_slice)
                event = strategy.event_proxy(hist)
                absolute[ticker] = MOMENTUM_WEIGHT * momentum + EVENT_WEIGHT * event
                raw_momentum[ticker] = _momentum_raw(hist, spy_slice)
                raw_momentum_no_spy[ticker] = _momentum_raw(hist, None)
                event_values[ticker] = event
            except Exception:
                continue

        def percentile_rank(values: dict[str, float]) -> pd.Series:
            series = pd.Series(values)
            if len(series) <= 1:
                return pd.Series(0.5, index=series.index)
            return (series.rank(method="average") - 1) / (len(series) - 1)

        ranks = percentile_rank(raw_momentum)
        live_ranks = percentile_rank(raw_momentum_no_spy)
        ranked = {
            ticker: MOMENTUM_WEIGHT * float(ranks[ticker])
            + EVENT_WEIGHT * event_values[ticker]
            for ticker in ranks.index
        }
        ranked_live = {
            ticker: MOMENTUM_WEIGHT * float(live_ranks[ticker])
            + EVENT_WEIGHT * event_values[ticker]
            for ticker in live_ranks.index
        }
        absolute_rows.append({"date": day, **absolute})
        ranked_rows.append({"date": day, **ranked})
        ranked_live_rows.append({"date": day, **ranked_live})
        if number % 100 == 0:
            print(f"Scored {number}/{len(days)} days", flush=True)

    return {
        "absolute": pd.DataFrame(absolute_rows).set_index("date"),
        "ranked": pd.DataFrame(ranked_rows).set_index("date"),
        "ranked_live": pd.DataFrame(ranked_live_rows).set_index("date"),
    }


def _bullish(spy: pd.Series, day: pd.Timestamp, sma_days: int) -> bool:
    close = spy.loc[:day].dropna()
    if len(close) < sma_days:
        return True
    sma = float(close.tail(sma_days).mean())
    return float(close.iloc[-1]) > sma * (1 - signals.REGIME_BUFFER_PCT)


def _target_weights(
    tickers: list[str],
    data: dict[str, pd.DataFrame],
    day: pd.Timestamp,
    variant: Variant,
) -> dict[str, float]:
    if not tickers:
        return {}
    if variant.inverse_vol:
        inverse = {ticker: 1 / _daily_vol(data[ticker], day) for ticker in tickers}
        total = sum(inverse.values())
        weights = {ticker: value / total for ticker, value in inverse.items()}
    else:
        weights = {ticker: 1 / len(tickers) for ticker in tickers}

    if variant.target_vol is not None:
        estimated = sqrt(sum(
            (weight * _daily_vol(data[ticker], day) * sqrt(252)) ** 2
            for ticker, weight in weights.items()
        ))
        scale = min(1.0, variant.target_vol / estimated) if estimated > 0 else 1.0
        weights = {ticker: weight * scale for ticker, weight in weights.items()}
    return weights


def _stop_price(
    ticker: str,
    entry_price: float,
    day: pd.Timestamp,
    data: dict[str, pd.DataFrame],
    variant: Variant,
) -> float | None:
    if variant.stop_mode == "none":
        return None
    distance = variant.stop_pct
    if variant.stop_mode == "atr":
        distance = max(distance, variant.atr_multiple * _atr_pct(data[ticker], day))
    return entry_price * (1 - distance)


def run_variant(
    variant: Variant,
    universe: list[str],
    data: dict[str, pd.DataFrame],
    days: pd.DatetimeIndex,
    scores: dict[str, pd.DataFrame],
    spy: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float = 10_000,
    cost_per_side: float = 0.001,
) -> tuple[pd.DataFrame, dict]:
    run_days = days[(days >= start) & (days <= end)]
    score_frame = scores[variant.score_mode]
    spy = spy.copy()
    spy.index = pd.to_datetime(spy.index)
    capital = initial_capital
    holdings: dict[str, Holding] = {}
    equity_rows = []
    switches = 0
    stop_exits = 0
    regime_exits = 0
    consecutive_bull = 0
    last_regime_exit = -999
    skip_entry_index: int | None = None

    for i in range(len(run_days) - 1):
        day, next_day = run_days[i], run_days[i + 1]
        if day not in score_frame.index:
            continue
        row = score_frame.loc[day].dropna().sort_values(ascending=False)
        if row.empty:
            continue

        today_bull = _bullish(spy, day, variant.regime_sma) if variant.regime else True
        consecutive_bull = consecutive_bull + 1 if today_bull else 0
        bull_confirmed = (
            today_bull
            and (not variant.regime or consecutive_bull >= bt.REGIME_CONFIRM_DAYS)
        )

        if variant.regime and not bull_confirmed and holdings:
            capital *= 1 - cost_per_side * sum(h.weight for h in holdings.values())
            holdings = {}
            switches += 1
            regime_exits += 1
            last_regime_exit = i

        in_cooldown = variant.regime and i - last_regime_exit < bt.REGIME_REENTRY_COOLDOWN
        may_enter = bull_confirmed and not in_cooldown and skip_entry_index != i

        action_taken = False
        if holdings and may_enter and len(holdings) < variant.positions:
            missing = variant.positions - len(holdings)
            additions = [ticker for ticker in row.index if ticker not in holdings][:missing]
            valid_additions = [
                ticker for ticker in additions
                if _px(data[ticker], day, "Close") is not None
            ]
            if valid_additions:
                new_tickers = list(holdings) + valid_additions
                new_weights = _target_weights(new_tickers, data, day, variant)
                old_weights = {ticker: h.weight for ticker, h in holdings.items()}
                turnover = sum(abs(
                    new_weights.get(ticker, 0.0) - old_weights.get(ticker, 0.0)
                ) for ticker in set(new_weights) | set(old_weights))
                capital *= 1 - cost_per_side * turnover
                for ticker in valid_additions:
                    entry_price = _px(data[ticker], day, "Close")
                    holdings[ticker] = Holding(
                        entry_price=entry_price,
                        entry_date=day,
                        held_days=0,
                        stop_price=_stop_price(ticker, entry_price, day, data, variant),
                        weight=new_weights[ticker],
                    )
                for ticker in holdings:
                    holdings[ticker].weight = new_weights[ticker]
                switches += 1
                action_taken = True

        if holdings and may_enter and not action_taken:
            current = list(holdings)
            outsiders = [ticker for ticker in row.index if ticker not in holdings]
            if outsiders:
                candidate = outsiders[0]
                weakest = min(current, key=lambda ticker: float(row.get(ticker, 0.5)))
                edge = float(row[candidate]) - float(row.get(weakest, 0.5))
                held_long_enough = holdings[weakest].held_days >= variant.min_hold
                fast = (
                    variant.fast_threshold is not None
                    and edge >= variant.fast_threshold
                )
                entry_price = _px(data[candidate], day, "Close")
                if (
                    edge > variant.threshold
                    and (held_long_enough or fast)
                    and entry_price is not None
                ):
                    new_tickers = [ticker for ticker in current if ticker != weakest]
                    new_tickers.append(candidate)
                    new_weights = _target_weights(new_tickers, data, day, variant)
                    old_weights = {ticker: h.weight for ticker, h in holdings.items()}
                    turnover = sum(abs(
                        new_weights.get(ticker, 0.0) - old_weights.get(ticker, 0.0)
                    ) for ticker in set(new_weights) | set(old_weights))
                    capital *= 1 - cost_per_side * turnover
                    del holdings[weakest]
                    holdings[candidate] = Holding(
                        entry_price=entry_price,
                        entry_date=day,
                        held_days=0,
                        stop_price=_stop_price(
                            candidate, entry_price, day, data, variant
                        ),
                        weight=new_weights[candidate],
                    )
                    for ticker in holdings:
                        holdings[ticker].weight = new_weights[ticker]
                    switches += 1

        elif not holdings and may_enter:
            selected = list(row.index[: variant.positions])
            weights = _target_weights(selected, data, day, variant)
            for ticker in selected:
                entry_price = _px(data[ticker], day, "Close")
                if entry_price:
                    holdings[ticker] = Holding(
                        entry_price=entry_price,
                        entry_date=day,
                        held_days=0,
                        stop_price=_stop_price(ticker, entry_price, day, data, variant),
                        weight=weights[ticker],
                    )
            if holdings:
                capital *= 1 - cost_per_side * sum(h.weight for h in holdings.values())
                switches += 1

        portfolio_return = 0.0
        stopped = []
        for ticker, holding in holdings.items():
            close_day = _px(data[ticker], day, "Close")
            close_next = _px(data[ticker], next_day, "Close")
            if close_day is None or close_next is None:
                continue
            asset_return = close_next / close_day - 1
            if holding.stop_price is not None:
                trigger = False
                fill = close_next
                if variant.stop_mode == "close":
                    trigger = close_next <= holding.stop_price
                else:
                    low_next = _px(data[ticker], next_day, "Low")
                    open_next = _px(data[ticker], next_day, "Open")
                    trigger = low_next is not None and low_next <= holding.stop_price
                    if trigger:
                        fill = (
                            open_next
                            if open_next is not None and open_next <= holding.stop_price
                            else holding.stop_price
                        )
                if trigger:
                    asset_return = fill / close_day - 1
                    stopped.append(ticker)
            portfolio_return += holding.weight * asset_return

        capital *= 1 + portfolio_return
        if stopped:
            sold_weight = sum(holdings[ticker].weight for ticker in stopped)
            capital *= 1 - cost_per_side * sold_weight
            for ticker in stopped:
                del holdings[ticker]
            stop_exits += len(stopped)
            switches += 1
            skip_entry_index = i + 1

        for holding in holdings.values():
            holding.held_days += 1

        equity_rows.append({
            "date": next_day,
            "value": capital,
            "daily_return": portfolio_return,
            "exposure": sum(h.weight for h in holdings.values()),
        })

    equity = pd.DataFrame(equity_rows).set_index("date")
    returns = equity["value"].pct_change().dropna()
    total_return = float(equity["value"].iloc[-1] / initial_capital - 1)
    annual_return = (1 + total_return) ** (252 / max(len(equity), 1)) - 1
    annual_vol = float(returns.std() * sqrt(252)) if len(returns) > 1 else 0.0
    sharpe = (
        (float(returns.mean()) * 252 - 0.05) / annual_vol
        if annual_vol > 0 else np.nan
    )
    drawdown = equity["value"] / equity["value"].cummax() - 1
    metrics = {
        "variant": variant.name,
        "period_start": start.date(),
        "period_end": end.date(),
        "total_return_pct": total_return * 100,
        "annual_return_pct": annual_return * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": float(drawdown.min()) * 100,
        "annual_vol_pct": annual_vol * 100,
        "switches": switches,
        "stop_exits": stop_exits,
        "regime_exits": regime_exits,
        "average_exposure_pct": float(equity["exposure"].mean()) * 100,
    }
    return equity, metrics


def variants() -> list[Variant]:
    base = Variant("baseline_abs_close_stop")
    return [
        base,
        Variant("current_live_rank_no_spy", score_mode="ranked_live"),
        Variant(
            "current_live_intraday_stop6",
            score_mode="ranked_live",
            stop_mode="intraday",
        ),
        Variant(
            "live_no_spy_hold1_threshold016",
            score_mode="ranked_live",
            min_hold=1,
            threshold=0.16,
        ),
        Variant(
            "live_h1_t016_intraday_stop6",
            score_mode="ranked_live",
            min_hold=1,
            threshold=0.16,
            stop_mode="intraday",
        ),
        Variant(
            "live_h1_t016_atr15",
            score_mode="ranked_live",
            min_hold=1,
            threshold=0.16,
            stop_mode="atr",
            atr_multiple=1.5,
        ),
        Variant(
            "live_h1_t016_atr20",
            score_mode="ranked_live",
            min_hold=1,
            threshold=0.16,
            stop_mode="atr",
            atr_multiple=2.0,
        ),
        Variant(
            "live_h1_t016_no_stop",
            score_mode="ranked_live",
            min_hold=1,
            threshold=0.16,
            stop_mode="none",
        ),
        Variant(
            "live_no_spy_hold3_threshold012",
            score_mode="ranked_live",
            min_hold=3,
            threshold=0.12,
        ),
        Variant("ranked_score" , score_mode="ranked"),
        Variant("ranked_hold1", score_mode="ranked", min_hold=1),
        Variant("ranked_hold3", score_mode="ranked", min_hold=3),
        Variant("ranked_threshold_012", score_mode="ranked", threshold=0.12),
        Variant("ranked_threshold_024", score_mode="ranked", threshold=0.24),
        Variant("ranked_regime100", score_mode="ranked", regime_sma=100),
        Variant("ranked_fast030", score_mode="ranked", fast_threshold=0.30),
        Variant("ranked_no_stop", score_mode="ranked", stop_mode="none"),
        Variant("ranked_intraday_stop_6", score_mode="ranked", stop_mode="intraday"),
        Variant(
            "ranked_atr_stop_15",
            score_mode="ranked",
            stop_mode="atr",
            atr_multiple=1.5,
        ),
        Variant(
            "ranked_atr_stop_20",
            score_mode="ranked",
            stop_mode="atr",
            atr_multiple=2.0,
        ),
        Variant("ranked_vol_target_18", score_mode="ranked", target_vol=0.18),
        Variant("no_regime", regime=False),
        Variant("regime_sma100", regime_sma=100),
        Variant("regime_sma200", regime_sma=200),
        Variant("fast_switch_030", fast_threshold=0.30),
        Variant("fast_switch_035", fast_threshold=0.35),
        Variant("fast_switch_040", fast_threshold=0.40),
        Variant("vol_target_12", target_vol=0.12),
        Variant("vol_target_15", target_vol=0.15),
        Variant("vol_target_18", target_vol=0.18),
        Variant("top2_equal", positions=2),
        Variant("top2_inverse_vol", positions=2, inverse_vol=True),
        Variant("no_stop", stop_mode="none"),
        Variant("intraday_stop_6", stop_mode="intraday"),
        Variant("atr_stop_15", stop_mode="atr", atr_multiple=1.5),
        Variant("atr_stop_20", stop_mode="atr", atr_multiple=2.0),
        Variant(
            "combo_fast035_vol15_intraday",
            fast_threshold=0.35,
            target_vol=0.15,
            stop_mode="intraday",
        ),
        Variant(
            "combo_top2_inv_regime100",
            positions=2,
            inverse_vol=True,
            regime_sma=100,
            stop_mode="intraday",
        ),
    ]


def print_robustness_tables(
    universe: list[str],
    data: dict[str, pd.DataFrame],
    days: pd.DatetimeIndex,
    scores: dict[str, pd.DataFrame],
    spy: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    recent_start: pd.Timestamp,
    cost_per_side: float,
) -> None:
    grid_rows = []
    for score_mode in ("ranked_live", "ranked"):
        for hold in (1, 2, 3):
            for threshold in (0.08, 0.12, 0.16, 0.20, 0.24, 0.28):
                variant = Variant(
                    f"grid_{score_mode}_h{hold}_t{threshold:.2f}",
                    score_mode=score_mode,
                    min_hold=hold,
                    threshold=threshold,
                )
                for label, period_start in (("full", start), ("recent", recent_start)):
                    _, metrics = run_variant(
                        variant,
                        universe,
                        data,
                        days,
                        scores,
                        spy,
                        period_start,
                        end,
                        cost_per_side=cost_per_side,
                    )
                    grid_rows.append({
                        "score_mode": score_mode,
                        "hold": hold,
                        "threshold": threshold,
                        "period": label,
                        "return": metrics["total_return_pct"],
                        "sharpe": metrics["sharpe"],
                        "drawdown": metrics["max_drawdown_pct"],
                    })
    grid = pd.DataFrame(grid_rows)
    for score_mode in ("ranked_live", "ranked"):
        for period in ("full", "recent"):
            subset = grid[
                (grid.period == period) & (grid.score_mode == score_mode)
            ]
            label = "LIVE NO-SPY" if score_mode == "ranked_live" else "WITH-SPY"
            print(f"\n{period.upper()} {label} GRID: SHARPE")
            print(subset.pivot(index="hold", columns="threshold", values="sharpe")
                  .to_string(float_format=lambda value: f"{value:.3f}"))
            print(f"\n{period.upper()} {label} GRID: TOTAL RETURN %")
            print(subset.pivot(index="hold", columns="threshold", values="return")
                  .to_string(float_format=lambda value: f"{value:.1f}"))

    candidates = [
        Variant("baseline_abs", score_mode="absolute"),
        Variant("live_h2_t018", score_mode="ranked_live"),
        Variant(
            "live_h1_t016",
            score_mode="ranked_live",
            min_hold=1,
            threshold=0.16,
        ),
        Variant(
            "live_h1_t016_intraday",
            score_mode="ranked_live",
            min_hold=1,
            threshold=0.16,
            stop_mode="intraday",
        ),
        Variant(
            "live_h1_t016_atr20",
            score_mode="ranked_live",
            min_hold=1,
            threshold=0.16,
            stop_mode="atr",
            atr_multiple=2.0,
        ),
        Variant(
            "live_h3_t012",
            score_mode="ranked_live",
            min_hold=3,
            threshold=0.12,
        ),
        Variant("ranked_h2_t018", score_mode="ranked"),
        Variant("ranked_h2_t012", score_mode="ranked", threshold=0.12),
        Variant("ranked_h1_t012", score_mode="ranked", min_hold=1, threshold=0.12),
        Variant("ranked_h3_t024", score_mode="ranked", min_hold=3, threshold=0.24),
        Variant(
            "ranked_atr20",
            score_mode="ranked",
            stop_mode="atr",
            atr_multiple=2.0,
        ),
    ]
    annual_rows = []
    for variant in candidates:
        equity, _ = run_variant(
            variant,
            universe,
            data,
            days,
            scores,
            spy,
            start,
            end,
            cost_per_side=cost_per_side,
        )
        daily = equity["value"].pct_change().dropna()
        for year, yearly in daily.groupby(daily.index.year):
            curve = (1 + yearly).cumprod()
            volatility = float(yearly.std() * sqrt(252))
            annual_rows.append({
                "variant": variant.name,
                "year": year,
                "return": float(curve.iloc[-1] - 1) * 100,
                "sharpe": (
                    (float(yearly.mean()) * 252 - 0.05) / volatility
                    if volatility > 0 else np.nan
                ),
                "drawdown": float((curve / curve.cummax() - 1).min()) * 100,
            })
    annual = pd.DataFrame(annual_rows)
    print("\nCALENDAR-YEAR RETURNS % (CONTINUOUS PORTFOLIO)")
    print(annual.pivot(index="variant", columns="year", values="return")
          .to_string(float_format=lambda value: f"{value:.1f}"))
    print("\nCALENDAR-YEAR SHARPE")
    print(annual.pivot(index="variant", columns="year", values="sharpe")
          .to_string(float_format=lambda value: f"{value:.2f}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-09-14")
    parser.add_argument("--end", default="2026-09-11")
    parser.add_argument("--recent-start", default="2024-09-14")
    parser.add_argument("--max-universe", type=int, default=150)
    parser.add_argument("--cost-per-side", type=float, default=0.001)
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument(
        "--cache-file",
        default="/private/tmp/relay_strategy_variants_cache.pkl",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    cache_key = (CACHE_VERSION, args.start, args.end, args.max_universe)
    cache_path = Path(args.cache_file)
    cached = None
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                candidate = pickle.load(handle)
            if candidate.get("key") == cache_key:
                cached = candidate
        except Exception:
            cached = None
    if cached is not None:
        universe = cached["universe"]
        data = cached["data"]
        spy = cached["spy"]
        days = cached["days"]
        scores = cached["scores"]
        print(f"Loaded score cache: {cache_path}", flush=True)
    else:
        raw_universe = bt.get_universe()
        all_data = bt.download_history(raw_universe, start, end)
        universe = bt.rank_by_liquidity(all_data, args.max_universe)
        data = {ticker: all_data[ticker] for ticker in universe}
        spy = bt.download_spy(start, end)
        days = pd.DatetimeIndex(sorted({
            day
            for frame in data.values()
            for day in frame.index
            if pd.Timestamp(start) <= day <= pd.Timestamp(end)
        }))
        scores = build_score_matrices(universe, data, days, spy)
        with cache_path.open("wb") as handle:
            pickle.dump({
                "key": cache_key,
                "universe": universe,
                "data": data,
                "spy": spy,
                "days": days,
                "scores": scores,
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved score cache: {cache_path}", flush=True)
    print(
        "Index diagnostics:",
        f"days={len(days)} {days.dtype}",
        f"scores={len(scores['absolute'])} {scores['absolute'].index.dtype}",
        f"first_day={days[0] if len(days) else None}",
        f"first_score={scores['absolute'].index[0] if len(scores['absolute']) else None}",
        flush=True,
    )

    periods = {
        "full": (pd.Timestamp(args.start), pd.Timestamp(args.end)),
        "recent": (pd.Timestamp(args.recent_start), pd.Timestamp(args.end)),
    }
    rows = []
    for variant in variants():
        for label, (period_start, period_end) in periods.items():
            _, metrics = run_variant(
                variant,
                universe,
                data,
                days,
                scores,
                spy,
                period_start,
                period_end,
                cost_per_side=args.cost_per_side,
            )
            metrics["period"] = label
            rows.append(metrics)
            print(
                f"{variant.name:34s} {label:6s} "
                f"ret={metrics['total_return_pct']:8.2f}% "
                f"sharpe={metrics['sharpe']:6.3f} "
                f"dd={metrics['max_drawdown_pct']:7.2f}%",
                flush=True,
            )

    result = pd.DataFrame(rows)
    print("\nFULL PERIOD RANKED BY SHARPE")
    columns = [
        "variant", "total_return_pct", "annual_return_pct", "sharpe",
        "max_drawdown_pct", "annual_vol_pct", "switches", "stop_exits",
        "average_exposure_pct",
    ]
    print(
        result[result.period == "full"][columns]
        .sort_values("sharpe", ascending=False)
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print("\nRECENT PERIOD RANKED BY SHARPE")
    print(
        result[result.period == "recent"][columns]
        .sort_values("sharpe", ascending=False)
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    if not args.skip_robustness:
        print_robustness_tables(
            universe,
            data,
            days,
            scores,
            spy,
            pd.Timestamp(args.start),
            pd.Timestamp(args.end),
            pd.Timestamp(args.recent_start),
            args.cost_per_side,
        )


if __name__ == "__main__":
    main()
