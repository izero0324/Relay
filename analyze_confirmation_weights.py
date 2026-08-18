"""Compare switch-confirmation days and momentum/volume weights.

The historical engine has no point-in-time Event or Flow dataset, so those
signals are neutral constants and cannot be optimized honestly here.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import logging

import numpy as np
import pandas as pd

import backtest as bt
from signals import momentum_raw, rank_pct, volume_score
from universe import get_universe


def metrics(equity: pd.DataFrame, trades: pd.DataFrame, capital: float) -> dict:
    values = equity["portfolio_value"].astype(float)
    daily = values.pct_change().dropna()
    years = max(len(values) / 252.0, 1 / 252.0)
    total = values.iloc[-1] / capital - 1
    cagr = (values.iloc[-1] / capital) ** (1 / years) - 1
    dd = values / values.cummax() - 1
    sharpe = np.sqrt(252) * daily.mean() / daily.std() if daily.std() else 0.0
    return {
        "total_return_pct": round(total * 100, 3),
        "cagr_pct": round(cagr * 100, 3),
        "max_drawdown_pct": round(dd.min() * 100, 3),
        "sharpe": round(float(sharpe), 3),
        "completed_trades": int(len(trades)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-08-18")
    parser.add_argument("--end", default="2026-08-17")
    parser.add_argument("--max-universe", type=int, default=150)
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--output", default="confirmation_weight_results.json")
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.WARNING)
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    raw = get_universe()
    all_data = bt.download_history(raw, start, end)
    universe = bt.rank_by_liquidity(all_data, args.max_universe)
    data = {ticker: all_data[ticker] for ticker in universe}
    spy = bt.download_spy(start, end)
    all_dates = sorted({d for frame in data.values() for d in frame.index
                        if pd.Timestamp(start) <= d <= pd.Timestamp(end)})
    days = pd.DatetimeIndex(all_dates)

    # Cache the expensive, weight-independent signal calculations once.
    component_cache: dict[pd.Timestamp, tuple[dict, dict]] = {}
    for i, prev_day in enumerate(days[:-1], 1):
        spy_slice = spy.loc[:prev_day] if spy is not None else None
        raws, volumes = {}, {}
        for ticker in universe:
            hist = data[ticker].loc[:prev_day]
            if len(hist) < bt.MIN_HISTORY_ROWS:
                continue
            try:
                raws[ticker] = momentum_raw(hist, spy_slice=spy_slice)
                volumes[ticker] = volume_score(hist)
            except Exception:
                continue
        component_cache[prev_day] = (rank_pct(raws), volumes)
        if i % 100 == 0:
            print(f"precomputed {i}/{len(days)-1} days", flush=True)

    original_score = bt.score_universe
    active_momentum_weight = 0.5

    def cached_score(_universe, _data, prev_day, _spy_slice, **_kwargs):
        momentum, volume = component_cache.get(prev_day, ({}, {}))
        wm = active_momentum_weight
        return {ticker: wm * score + (1 - wm) * volume[ticker]
                for ticker, score in momentum.items() if ticker in volume}

    bt.score_universe = cached_score
    split = days[int(len(days) * 2 / 3)]
    rows = []
    try:
        for confirm in (1, 2, 3):
            bt.SWITCH_CONFIRM_DAYS = confirm
            for wm in np.linspace(0, 1, 11):
                active_momentum_weight = float(wm)
                for sample, sample_days in (
                    ("train", days[days < split]),
                    ("test", days[days >= split]),
                    ("full", days),
                ):
                    equity, trades = bt.run_simulation(
                        universe, data, sample_days, args.capital,
                        bt.SWITCH_THRESHOLD, spy,
                    )
                    row = {"sample": sample, "confirm_days": confirm,
                           "momentum_weight": round(float(wm), 2),
                           "volume_weight": round(1 - float(wm), 2)}
                    row.update(metrics(equity, trades, args.capital))
                    rows.append(row)
            print(f"finished confirmation={confirm}", flush=True)
    finally:
        bt.score_universe = original_score

    # Select only on train; evaluate the chosen setting on untouched test data.
    frame = pd.DataFrame(rows)
    train = frame[frame["sample"] == "train"].copy()
    train["selection_score"] = train["sharpe"] + train["cagr_pct"] / 100
    chosen = train.sort_values("selection_score", ascending=False).iloc[0]
    selected_test = frame[
        (frame["sample"] == "test")
        & (frame["confirm_days"] == chosen["confirm_days"])
        & (frame["momentum_weight"] == chosen["momentum_weight"])
    ].iloc[0]
    payload = {
        "period": {"start": args.start, "end": args.end,
                   "train_end_exclusive": str(split.date()),
                   "universe_size": len(universe)},
        "selection_rule": "maximize train Sharpe + train CAGR",
        "selected_train": chosen.to_dict(),
        "selected_test": selected_test.to_dict(),
        "results": rows,
    }
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({"selected_train": payload["selected_train"],
                      "selected_test": payload["selected_test"]}, indent=2))


if __name__ == "__main__":
    main()
