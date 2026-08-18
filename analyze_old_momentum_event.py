"""Run the legacy close-execution backtest with Momentum + Event proxy only.

This intentionally preserves the reference version's execution, regime,
cost, and switching mechanics. Historical Yahoo news snapshots do not exist,
so Event is represented by a point-in-time price-reaction/volume proxy.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np


REFERENCE = "/Users/andrewyang/Desktop/workspace/trading212/daily_scanner"
sys.path.insert(0, REFERENCE)
legacy = importlib.import_module("backtest")
legacy_signals = importlib.import_module("signals")


MOMENTUM_WEIGHT = 0.40 / (0.40 + 0.25)
EVENT_WEIGHT = 0.25 / (0.40 + 0.25)


def event_proxy(hist):
    """Backtest-safe directional event reaction confirmed by relative volume."""
    window = legacy.VOLUME_WINDOW_DAYS
    if len(hist) < window + 2:
        return 0.5
    close = hist["Close"].squeeze()
    volume = hist["Volume"].squeeze()
    previous = float(close.iloc[-2])
    baseline = float(volume.iloc[-(window + 1):-1].mean())
    if previous <= 0 or baseline <= 0:
        return 0.5
    move = float(close.iloc[-1]) / previous - 1.0
    relative_volume = max(float(volume.iloc[-1]) / baseline, 0.01)
    confirmation = min(max(np.log(relative_volume) + 1.0, 0.0), 2.0) / 2.0
    impact = np.tanh(move * 18.0) * confirmation
    return float(np.clip(0.5 + 0.5 * impact, 0.0, 1.0))


def momentum_event_score(hist, spy_slice=None):
    momentum = legacy_signals.momentum_score(hist, spy_slice=spy_slice)
    event = event_proxy(hist)
    return MOMENTUM_WEIGHT * momentum + EVENT_WEIGHT * event


legacy.backtest_composite_score = momentum_event_score


if __name__ == "__main__":
    print(
        f"Legacy close execution; Momentum={MOMENTUM_WEIGHT:.4f}, "
        f"EventProxy={EVENT_WEIGHT:.4f}"
    )
    legacy.main()
