"""
signals.py — Compute the four orthogonal signal categories.

Each function returns a float in [0.0, 1.0]:
  0.0  =  strongly bearish / no signal
  0.5  =  neutral
  1.0  =  strongly bullish

v2 improvements:
  4.1  momentum_score  — now risk-adjusted + relative strength vs SPY
  4.2  volume_score    — unchanged (volume confirmation)
  4.3  event_score     — unchanged (earnings + news sentiment)
  4.4  flow_score      — unchanged (pre-market gap + market context)
  NEW  regime_check    — returns bool: True = SPY in uptrend, safe to be long
"""

import logging
from datetime import date, datetime, timedelta
from math import sqrt

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional, Union

from config import (
    EARNINGS_RECENT_DAYS,
    EARNINGS_UPCOMING_DAYS,
    HIGH_WINDOW_DAYS,
    MIN_HOLD_DAYS,
    MOMENTUM_MEDIUM_DAYS,
    REGIME_BUFFER_PCT,
    REGIME_FILTER_ENABLED,
    REGIME_SMA_DAYS,
    RELATIVE_STRENGTH_ENABLED,
    RISK_ADJUSTED_MOMENTUM,
    VOL_PENALTY_DAYS,
    VOL_PENALTY_ENABLED,
    VOL_PENALTY_STRENGTH,
    VOL_PENALTY_THRESHOLD,
    VOLUME_SPIKE_MULT,
    VOLUME_WINDOW_DAYS,
)

logger = logging.getLogger(__name__)


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _sigmoid(x: float, k: float = 5.0) -> float:
    return float(1.0 / (1.0 + np.exp(-k * x)))

def _clip01(x: float) -> float:
    if np.isnan(x):
        return 0.5  # neutral — np.clip would propagate the NaN
    return float(np.clip(x, 0.0, 1.0))


# ── Regime check ──────────────────────────────────────────────────────────────

def regime_is_bullish(spy_hist: Union[pd.DataFrame, pd.Series]) -> bool:
    """
    Returns True if SPY is in a bullish regime (above its N-day SMA).
    When False, the backtest/scanner goes to cash — no longs permitted.

    Args:
        spy_hist: SPY OHLCV DataFrame or Close price Series.
                  Must have at least REGIME_SMA_DAYS rows.
    """
    if not REGIME_FILTER_ENABLED:
        return True

    try:
        if isinstance(spy_hist, pd.DataFrame):
            close = spy_hist["Close"].squeeze().dropna()
        else:
            close = spy_hist.dropna()

        if len(close) < REGIME_SMA_DAYS:
            logger.debug("Not enough SPY history for regime check — defaulting bullish")
            return True

        sma   = float(close.iloc[-REGIME_SMA_DAYS:].mean())
        price = float(close.iloc[-1])
        # Allow a small buffer: price must be > SMA * (1 - buffer)
        threshold = sma * (1 - REGIME_BUFFER_PCT)
        is_bull   = price > threshold

        direction = "BULL ▲" if is_bull else "BEAR ▼"
        logger.debug(
            f"Regime: SPY {price:.2f} vs SMA{REGIME_SMA_DAYS} {sma:.2f} → {direction}"
        )
        return is_bull

    except Exception as e:
        logger.warning(f"Regime check failed ({e}) — defaulting to bullish")
        return True


# ── 4.1 Momentum Signal (v2) ──────────────────────────────────────────────────

def momentum_raw(
    hist: pd.DataFrame,
    spy_slice: Optional[pd.Series] = None,
) -> float:
    """
    Risk-adjusted, relative-strength momentum — UNCLIPPED raw value.

    Same composite as momentum_score but without the final [0, 1] clip,
    so strong names don't all saturate at 1.0. Use with rank_pct() for
    cross-sectional ranking; use momentum_score() for an absolute score.

    Components:
      1. Risk-adjusted 1-day return  (return / sqrt(recent_daily_vol))
      2. Risk-adjusted 5-day return
      3. 20-day high proximity
      4. Breakout bonus (+0.15 if today broke above prior 20-day high)
      5. Relative strength vs SPY (-/+ adjustment if spy_slice provided)
      6. Trend confirmation: above 10-day MA → small bonus
      7. Volatility penalty: penalises very high recent daily vol

    Args:
        hist      : OHLCV DataFrame (needs ≥ HIGH_WINDOW_DAYS + MOMENTUM_MEDIUM_DAYS rows)
        spy_slice : SPY Close Series aligned to same dates (optional)

    Returns:
        float in [0, 1]
    """
    min_rows = HIGH_WINDOW_DAYS + MOMENTUM_MEDIUM_DAYS + 2
    if len(hist) < min_rows:
        return 0.5

    close = hist["Close"].squeeze()
    high  = hist["High"].squeeze()

    # ── Raw returns ───────────────────────────────────────────────────────────
    r1 = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2])
    r5 = (
        float((close.iloc[-1] - close.iloc[-(MOMENTUM_MEDIUM_DAYS + 1)]) /
              close.iloc[-(MOMENTUM_MEDIUM_DAYS + 1)])
        if len(close) >= MOMENTUM_MEDIUM_DAYS + 1 else 0.0
    )

    # ── Recent volatility ─────────────────────────────────────────────────────
    recent_rets = close.pct_change().iloc[-VOL_PENALTY_DAYS:].dropna()
    recent_vol  = float(recent_rets.std()) if len(recent_rets) > 1 else 0.02

    # ── Risk adjustment ───────────────────────────────────────────────────────
    if RISK_ADJUSTED_MOMENTUM and recent_vol > 0:
        # Normalise to "how many daily vol units did we move?"
        r1_adj = r1 / recent_vol
        r5_adj = (r5 / (recent_vol * sqrt(MOMENTUM_MEDIUM_DAYS)))
    else:
        r1_adj = r1 * 50    # same scale as original (sigmoid k=50)
        r5_adj = r5 * 20

    # ── Relative strength vs SPY ──────────────────────────────────────────────
    rs_adj = 0.0
    if RELATIVE_STRENGTH_ENABLED and spy_slice is not None and len(spy_slice) >= MOMENTUM_MEDIUM_DAYS + 1:
        try:
            spy_r1 = float((spy_slice.iloc[-1] - spy_slice.iloc[-2]) / spy_slice.iloc[-2])
            spy_r5 = float((spy_slice.iloc[-1] - spy_slice.iloc[-(MOMENTUM_MEDIUM_DAYS + 1)]) /
                           spy_slice.iloc[-(MOMENTUM_MEDIUM_DAYS + 1)])
            spy_vol = float(spy_slice.pct_change().iloc[-VOL_PENALTY_DAYS:].dropna().std())
            if spy_vol > 0:
                rel_r1 = (r1 - spy_r1) / spy_vol      # excess return in SPY vol units
                rel_r5 = (r5 - spy_r5) / (spy_vol * sqrt(MOMENTUM_MEDIUM_DAYS))
                rs_adj = (_sigmoid(rel_r1, k=1.5) - 0.5) * 0.20 + \
                         (_sigmoid(rel_r5, k=1.0) - 0.5) * 0.10
        except Exception:
            pass

    # ── 20-day high proximity ─────────────────────────────────────────────────
    high_20d     = float(high.iloc[-HIGH_WINDOW_DAYS:].max())
    pct_from_20h = (float(close.iloc[-1]) - high_20d) / high_20d

    # ── Breakout bonus ────────────────────────────────────────────────────────
    prior_high = float(high.iloc[-(HIGH_WINDOW_DAYS + 1):-1].max())
    breakout_bonus = 0.15 if float(close.iloc[-1]) > prior_high else 0.0

    # ── Trend confirmation (above 10-day MA) ──────────────────────────────────
    if len(close) >= 10:
        ma10 = float(close.iloc[-10:].mean())
        trend_bonus = 0.05 if float(close.iloc[-1]) > ma10 else -0.05
    else:
        trend_bonus = 0.0

    # ── Volatility penalty ────────────────────────────────────────────────────
    vol_penalty = 0.0
    if VOL_PENALTY_ENABLED and recent_vol > VOL_PENALTY_THRESHOLD:
        excess      = recent_vol - VOL_PENALTY_THRESHOLD
        vol_penalty = min(excess * VOL_PENALTY_STRENGTH, 0.25)

    # ── Combine ───────────────────────────────────────────────────────────────
    s_r1   = _sigmoid(r1_adj, k=1.5)   # k=1.5 because r1_adj already in vol units
    s_r5   = _sigmoid(r5_adj, k=1.0)
    s_high = _clip01(1.0 + pct_from_20h * 5.0)

    raw = (
        (s_r1 + s_r5 + s_high) / 3.0
        + breakout_bonus
        + trend_bonus
        + rs_adj
        - vol_penalty
    )
    if np.isnan(raw):
        return 0.5
    return float(raw)


def momentum_score(
    hist: pd.DataFrame,
    spy_slice: Optional[pd.Series] = None,
) -> float:
    """Absolute momentum score: momentum_raw clipped to [0, 1]."""
    return _clip01(momentum_raw(hist, spy_slice=spy_slice))


def rank_pct(raw: dict) -> dict:
    """
    Percentile-rank a {ticker: raw_value} dict into [0, 1].
    Ties share their average rank. A single entry maps to 0.5.
    """
    n = len(raw)
    if n == 0:
        return {}
    if n == 1:
        return {t: 0.5 for t in raw}
    s = pd.Series(raw)
    return ((s.rank(method="average") - 1) / (n - 1)).to_dict()


# ── 4.2 Volume Signal ─────────────────────────────────────────────────────────

def volume_score(hist: pd.DataFrame) -> float:
    """
    Detects abnormal volume vs 20-day baseline. Unchanged from v1.
    """
    if len(hist) < VOLUME_WINDOW_DAYS + 1:
        return 0.5

    volume    = hist["Volume"].squeeze()
    avg_vol   = float(volume.iloc[-(VOLUME_WINDOW_DAYS + 1):-1].mean())
    today_vol = float(volume.iloc[-1])

    if avg_vol <= 0:
        return 0.5

    vol_ratio   = today_vol / avg_vol
    spike_bonus = 0.15 if vol_ratio >= VOLUME_SPIKE_MULT else 0.0
    s_vol       = _sigmoid(np.log(max(vol_ratio, 0.01)), k=1.5)

    return _clip01(s_vol + spike_bonus)


# ── 4.3 Event / Information Signal ───────────────────────────────────────────

_POSITIVE_WORDS = {
    "beat", "beats", "record", "surge", "soar", "rally", "upgrade",
    "outperform", "raise", "raised", "bullish", "strong", "growth",
    "profit", "wins", "win", "approved", "approval", "launch", "buyback",
    "dividend", "acquisition", "partnership", "breakthrough", "exceeds",
}
_NEGATIVE_WORDS = {
    "miss", "misses", "cut", "downgrade", "underperform", "loss",
    "decline", "drop", "fall", "falls", "slump", "concern", "warning",
    "lawsuit", "recall", "investigation", "fraud", "suspend", "layoff",
    "layoffs", "bankruptcy", "default", "below", "disappoints", "weak",
}


def event_score(ticker_obj: yf.Ticker) -> float:
    """Earnings proximity + news sentiment. Unchanged from v1."""
    score = 0.50
    today = date.today()

    try:
        cal           = ticker_obj.calendar
        earnings_dates: list = []

        if isinstance(cal, dict):
            raw = cal.get("Earnings Date", [])
            earnings_dates = raw if isinstance(raw, (list, tuple)) else [raw]
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            col = None
            for c in cal.columns:
                if "earnings" in str(c).lower():
                    col = c
                    break
            if col:
                earnings_dates = cal[col].dropna().tolist()

        for ed in earnings_dates:
            if ed is None:
                continue
            if hasattr(ed, "date"):
                ed = ed.date()
            elif isinstance(ed, str):
                try:
                    ed = datetime.strptime(ed[:10], "%Y-%m-%d").date()
                except ValueError:
                    continue

            delta = (ed - today).days
            if -EARNINGS_RECENT_DAYS <= delta < 0:
                score += 0.30
            elif 0 <= delta <= EARNINGS_UPCOMING_DAYS:
                score += 0.25
            elif EARNINGS_UPCOMING_DAYS < delta <= 7:
                score += 0.10

    except Exception as e:
        logger.debug(f"Earnings calendar error: {e}")

    try:
        news_items = ticker_obj.news or []
        cutoff     = datetime.combine(today - timedelta(days=3), datetime.min.time())
        recent: list[str] = []   # recent headline titles
        for n in news_items:
            # New yfinance format nests fields under "content"; fall back to
            # the legacy flat format (title / providerPublishTime).
            content = n.get("content") or n
            title   = content.get("title") or ""
            ts      = content.get("pubDate") or content.get("displayTime")
            when    = None
            if isinstance(ts, str):
                try:
                    when = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    pass
            elif n.get("providerPublishTime"):
                when = datetime.fromtimestamp(n["providerPublishTime"])
            if title and when is not None and when >= cutoff:
                recent.append(title)

        pos = neg = 0
        for title in recent[:12]:
            title = title.lower()
            pos += sum(1 for w in _POSITIVE_WORDS if w in title)
            neg += sum(1 for w in _NEGATIVE_WORDS if w in title)
        if pos + neg > 0:
            score += ((pos - neg) / (pos + neg)) * 0.20
    except Exception as e:
        logger.debug(f"News sentiment error: {e}")

    return _clip01(score)


# ── 4.4 Flow / Pre-market Signal ─────────────────────────────────────────────

_FLOW_STATS = {"attempts": 0, "gap_ok": 0}


def reset_flow_stats() -> None:
    _FLOW_STATS["attempts"] = 0
    _FLOW_STATS["gap_ok"]   = 0


def flow_stats() -> dict:
    return dict(_FLOW_STATS)


def flow_score(ticker_obj: yf.Ticker, spy_trend: float) -> float:
    """
    Pre-market/latest gap vs previous close + SPY trend context.

    fast_info has no pre-market price field, so the latest traded price
    (including pre/post-market) comes from 1-minute prepost history.
    Run pre-market → this is the pre-market gap; run intraday → today's move.

    period="2d", not "1d": Yahoo resolves 1d as the CURRENT session and
    returns nothing between sessions / in early pre-market ("possibly
    delisted" spam). 2d always has a last bar to price against.
    """
    score = 0.50
    _FLOW_STATS["attempts"] += 1

    try:
        intraday = ticker_obj.history(period="2d", interval="1m", prepost=True)
        latest   = (
            float(intraday["Close"].dropna().iloc[-1])
            if len(intraday) and intraday["Close"].notna().any() else None
        )
        fi         = ticker_obj.fast_info
        prev_close = (
            getattr(fi, "previous_close", None) or
            getattr(fi, "regular_market_previous_close", None)
        )
        if latest and prev_close and float(prev_close) > 0:
            gap_pct = (latest - float(prev_close)) / float(prev_close)
            score  += _sigmoid(gap_pct, k=40) - 0.5
            _FLOW_STATS["gap_ok"] += 1
    except Exception as e:
        logger.debug(f"Pre-market data error: {e}")

    market_adj = (_sigmoid(spy_trend, k=20) - 0.5) * 0.3
    score += market_adj

    return _clip01(score)
