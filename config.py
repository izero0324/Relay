"""
config.py — Strategy configuration
All tunable parameters live here. No need to touch any other file.

v3 changes:
  - STOP_LOSS_PCT: per-trade hard stop at -6% to cap disasters
  - REGIME_CONFIRM_DAYS: SPY must be above SMA for N days before re-entry
  - REGIME_REENTRY_COOLDOWN: mandatory wait after a regime exit
  - BACKTEST_NORMALIZE_SCORE: fixes score ceiling bug (was 0.775 max, now 1.0)
  - MIN_MARKET_CAP_PROXY: raised MIN_AVG_VOLUME to block micro-caps like PSKY
"""

# ── Signal weights (must sum to 1.0) ──────────────────────────────────────────
WEIGHTS = {
    "momentum": 0.40,
    "volume":   0.15,
    "event":    0.25,
    "flow":     0.20,
}

# ── Decision threshold ────────────────────────────────────────────────────────
# Expressed in NORMALIZED momentum+volume units (the units the backtest
# validates). The live scanner scales it by the momentum+volume weight so a
# raw composite delta means the same thing in both places.
SWITCH_THRESHOLD = 0.18

# ── Switch confirmation ───────────────────────────────────────────────────────
# The edge must exceed the threshold on N consecutive scans/days before a
# switch fires. Filters one-day score noise (volume spikes etc.).
# Set to 1 to disable. Entries from CASH are never delayed.
SWITCH_CONFIRM_DAYS = 2

# ── Cross-sectional momentum ranking ──────────────────────────────────────────
# When True, the momentum component is the stock's percentile rank of raw
# (unclipped) momentum across the whole filtered universe that day, instead
# of an absolute sigmoid score. Removes the saturation at 1.0 that made
# ~30% of backtest picks tie-broken by liquidity order.
CROSS_SECTIONAL_RANK = True

# ── Scanner state file ────────────────────────────────────────────────────────
# Persists pending-switch confirmation state between daily live scans.
SCANNER_STATE_PATH = "scanner_state.json"

# ── Minimum hold period ───────────────────────────────────────────────────────
MIN_HOLD_DAYS = 2

# ── Per-trade stop loss ───────────────────────────────────────────────────────
# Exit position next day if cumulative return from entry drops below this.
# Set to None to disable.  -0.06 = -6% hard stop.
STOP_LOSS_PCT = -0.06

# ── Regime filter ─────────────────────────────────────────────────────────────
REGIME_FILTER_ENABLED = True
REGIME_SMA_DAYS       = 50     # SPY must be above this SMA to permit longs
REGIME_BUFFER_PCT     = 0.005  # raised from 0 → 0.5% buffer below SMA
                                # reduces flip-flopping near the SMA line

# NEW: SPY must have been above its SMA for at least N consecutive days
# before the strategy re-enters a long position after a bear regime.
# Prevents buying straight back into a bounce that fails.
REGIME_CONFIRM_DAYS   = 3

# NEW: Mandatory days in cash after any REGIME_EXIT before new entries allowed.
# Prevents immediate re-entry on a 1-day bounce.
REGIME_REENTRY_COOLDOWN = 5

# ── Backtest score normalisation ──────────────────────────────────────────────
# In backtest mode, event and flow signals are fixed at 0.5 (neutral).
# This means the maximum achievable composite score is:
#   0.40×1 + 0.15×1 + 0.25×0.5 + 0.20×0.5 = 0.775
# Every top stock hits this ceiling → can't differentiate top candidates.
#
# When True (recommended), the backtest score is renormalised using only
# the active signals (momentum + volume), scaled to their combined weight.
# This restores the full [0, 1] range for ranking purposes.
BACKTEST_NORMALIZE_SCORE = True

# ── Universe sources ──────────────────────────────────────────────────────────
UNIVERSE_SOURCES = ["sp500", "nasdaq100"]

# ── Liquidity filters ─────────────────────────────────────────────────────────
MIN_AVG_VOLUME = 2_000_000   # raised from 1M → 2M to block micro/small caps
MIN_PRICE      = 10.0        # raised from $5 → $10

# ── Lookback windows (days) ───────────────────────────────────────────────────
MOMENTUM_SHORT_DAYS  = 1
MOMENTUM_MEDIUM_DAYS = 5
HIGH_WINDOW_DAYS     = 20
VOLUME_WINDOW_DAYS   = 20
VOL_PENALTY_DAYS     = 10

# ── Volume spike threshold ────────────────────────────────────────────────────
VOLUME_SPIKE_MULT = 2.0

# ── Risk-adjusted momentum ────────────────────────────────────────────────────
RISK_ADJUSTED_MOMENTUM = True

# ── Relative strength ─────────────────────────────────────────────────────────
RELATIVE_STRENGTH_ENABLED = True

# ── Volatility penalty ────────────────────────────────────────────────────────
VOL_PENALTY_ENABLED    = True
VOL_PENALTY_THRESHOLD  = 0.025
VOL_PENALTY_STRENGTH   = 4.0

# ── Earnings proximity scoring windows ───────────────────────────────────────
EARNINGS_RECENT_DAYS   = 2
EARNINGS_UPCOMING_DAYS = 3

# ── Scan settings ─────────────────────────────────────────────────────────────
TOP_CANDIDATES   = 10
PRE_FILTER_TOP_N = 60

# ── File paths ────────────────────────────────────────────────────────────────
TRADE_LOG_PATH = "trade_log.csv"
