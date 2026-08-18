"""
scorer.py — Composite scoring and Hold/Switch decision.

Decision rule (from TRD Section 3):
  IF best_candidate_score > current_position_score + SWITCH_THRESHOLD:
      SWITCH
  ELSE:
      HOLD
"""

from __future__ import annotations

from config import (
    BACKTEST_ACTIVE_SIGNALS,
    BACKTEST_NORMALIZE_SCORE,
    SWITCH_THRESHOLD,
    WEIGHTS,
)


def effective_live_threshold() -> float:
    """
    SWITCH_THRESHOLD in raw live-composite units.

    The threshold is validated on a score normalized over the signals active
    in that backtest. Scale it by those signals' live composite weight so the
    live and backtest score deltas stay in the same units.
    """
    if BACKTEST_NORMALIZE_SCORE:
        active_weight = sum(WEIGHTS[name] for name in BACKTEST_ACTIVE_SIGNALS)
        return SWITCH_THRESHOLD * active_weight
    return SWITCH_THRESHOLD


def composite_score(
    momentum: float,
    volume: float,
    event: float,
    flow: float,
) -> float:
    """
    Weighted sum of the four signal categories.

    All inputs are expected in [0, 1].
    Returns a float in [0, 1].
    """
    w = WEIGHTS
    return (
        w["momentum"] * momentum +
        w["volume"]   * volume   +
        w["event"]    * event    +
        w["flow"]     * flow
    )


def decision(
    current_ticker: str,
    current_score: float,
    best_ticker: str,
    best_score: float,
    threshold: float | None = None,
) -> tuple[str, str]:
    """
    Apply the Hold/Switch rule.

    Args:
        current_ticker : ticker symbol of the current holding (or "CASH")
        current_score  : composite score of the current position
        best_ticker    : top-ranked candidate from the scan
        best_score     : composite score of that candidate
        threshold      : override; defaults to effective_live_threshold()

    Returns:
        (action, reason)
          action : "HOLD" or "SWITCH"
          reason : human-readable explanation for the trade log
    """
    thr  = effective_live_threshold() if threshold is None else threshold
    diff = best_score - current_score

    if diff > thr:
        action = "SWITCH"
        reason = (
            f"Best candidate {best_ticker} scores {best_score:.3f} vs "
            f"current {current_ticker} at {current_score:.3f} "
            f"(edge Δ={diff:+.3f} exceeds threshold {thr:.3f})"
        )
    else:
        action = "HOLD"
        reason = (
            f"Current {current_ticker} scores {current_score:.3f}; "
            f"best candidate {best_ticker} scores {best_score:.3f} "
            f"(Δ={diff:+.3f} ≤ threshold {thr:.3f} — hold advantage)"
        )

    return action, reason
