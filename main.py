"""
main.py — Daily Rotation Scanner
─────────────────────────────────
Run each morning before market open:

  python main.py                    # prompts for current ticker
  python main.py AAPL               # pass ticker as argument
  python main.py CASH               # use CASH as current position

Output:
  - Ranked candidate table printed to terminal
  - HOLD or SWITCH decision with explanation
  - Row appended to trade_log.csv
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import textwrap
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from config import (
    CROSS_SECTIONAL_RANK,
    HIGH_WINDOW_DAYS,
    MIN_AVG_VOLUME,
    MIN_HOLD_DAYS,
    MIN_PRICE,
    PRE_FILTER_TOP_N,
    SCANNER_STATE_PATH,
    STOP_LOSS_PCT,
    SWITCH_CONFIRM_DAYS,
    TOP_CANDIDATES,
    TRADE_LOG_PATH,
    VOLUME_WINDOW_DAYS,
    WEIGHTS,
)
from position_risk import (
    TRADE_LOG_FIELDS,
    EntryDetails,
    StopCheck,
    ensure_trade_log_schema,
    evaluate_stop,
    fetch_regular_session_snapshot,
    find_entry_details,
)
from scorer import composite_score, decision
from signals import (
    event_score,
    flow_score,
    flow_stats,
    momentum_raw,
    momentum_score,
    rank_pct,
    reset_flow_stats,
    volume_score,
)
from universe import get_universe

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# yfinance logs a misleading "possibly delisted" ERROR for every empty
# response (rate limits, no bars yet, transient hiccups). We handle and
# summarize those cases ourselves, so silence the per-ticker spam.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# How many historical days to download (buffer added for weekends/holidays)
_LOOKBACK = HIGH_WINDOW_DAYS + VOLUME_WINDOW_DAYS + 15


# ─────────────────────────────────────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────────────────────────────────────

def get_current_ticker() -> str:
    """Read current holding from CLI argument or interactive prompt."""
    if len(sys.argv) > 1:
        return sys.argv[1].upper().strip()
    raw = input("\n  Enter your current holding (ticker or CASH): ").strip().upper()
    return raw or "CASH"


# ─────────────────────────────────────────────────────────────────────────────
# Market context
# ─────────────────────────────────────────────────────────────────────────────

def get_spy_trend() -> float:
    """SPY 5-day return as decimal. Used in flow signal as market context."""
    try:
        spy = yf.download("SPY", period="15d", progress=False, auto_adjust=True)
        close = spy["Close"].squeeze().dropna()
        if len(close) < 6:
            return 0.0
        return float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6])
    except Exception as e:
        logger.warning(f"SPY trend unavailable ({e}) — defaulting to 0")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def bulk_download(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV for all tickers in one batch request.
    Returns dict of {ticker: DataFrame} for tickers with sufficient data.
    """
    logger.info(f"Bulk downloading OHLCV for {len(tickers)} tickers…")

    raw = yf.download(
        tickers,
        period=f"{_LOOKBACK}d",
        progress=True,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )

    result: dict[str, pd.DataFrame] = {}
    min_rows = VOLUME_WINDOW_DAYS + 5

    for t in tickers:
        try:
            # yfinance returns MultiIndex columns when >1 ticker
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[t].copy()

            df.dropna(how="all", inplace=True)
            # Pre-/early-market runs include a partial "today" row whose price
            # columns are NaN (volume is 0, so how="all" keeps it) — drop it,
            # otherwise every score downstream becomes NaN.
            if "Close" in df.columns:
                df.dropna(subset=["Close"], inplace=True)
            if len(df) >= min_rows:
                result[t] = df
        except Exception:
            pass  # ticker not in download (delisted, bad symbol, etc.)

    logger.info(f"Usable OHLCV data: {len(result)} tickers")
    return result


def apply_liquidity_filter(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Remove stocks below minimum average volume or price."""
    filtered = {}
    for t, df in data.items():
        try:
            avg_vol   = float(df["Volume"].iloc[-VOLUME_WINDOW_DAYS:].mean())
            avg_price = float(df["Close"].iloc[-VOLUME_WINDOW_DAYS:].mean())
            if avg_vol >= MIN_AVG_VOLUME and avg_price >= MIN_PRICE:
                filtered[t] = df
        except Exception:
            pass
    logger.info(f"After liquidity filter: {len(filtered)} tickers")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def price_volume_scores(data: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    """
    Momentum and volume components for every ticker in the filtered universe.

    With CROSS_SECTIONAL_RANK, momentum is the percentile rank of raw
    (unclipped) momentum across the whole universe — no saturation ties.
    """
    v_scores = {t: volume_score(df) for t, df in data.items()}
    if CROSS_SECTIONAL_RANK:
        m_scores = rank_pct({t: momentum_raw(df) for t, df in data.items()})
    else:
        m_scores = {t: momentum_score(df) for t, df in data.items()}
    return m_scores, v_scores


def full_score(ticker: str, momentum: float, volume: float, spy_trend: float) -> dict:
    """
    Full 4-signal composite score for a single ticker.
    Momentum/volume come from price_volume_scores(); this adds the
    per-ticker event and flow API calls.
    """
    obj = yf.Ticker(ticker)
    e   = event_score(obj) if WEIGHTS["event"] > 0 else 0.50
    f   = flow_score(obj, spy_trend) if WEIGHTS["flow"] > 0 else 0.50
    c   = composite_score(momentum, volume, e, f)

    return {
        "Ticker":    ticker,
        "Momentum":  round(momentum, 3),
        "Volume":    round(volume, 3),
        "Event":     round(e, 3),
        "Flow":      round(f, 3),
        "Score":     round(c, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def append_trade_log(
    current: str,
    action: str,
    switch_to: str,
    reason: str,
    entry: EntryDetails | None = None,
    session_low: float | None = None,
    stop_status: str = "",
    stop_checked_through: datetime | None = None,
) -> None:
    """Append one row, migrating older logs to the risk-aware schema."""
    is_new = not os.path.exists(TRADE_LOG_PATH)
    if not is_new:
        ensure_trade_log_schema(TRADE_LOG_PATH)
    fieldnames = TRADE_LOG_FIELDS
    if not is_new:
        with open(TRADE_LOG_PATH, newline="", encoding="utf-8") as existing:
            fieldnames = list(csv.DictReader(existing).fieldnames or TRADE_LOG_FIELDS)
    with open(TRADE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow({
            "Date": date.today().isoformat(),
            "Current_Position": current,
            "Action": action,
            "Switch_To": switch_to if action in {"SWITCH", "STOP_LOSS"} else "",
            "Reason": reason,
            "Outcome": "",
            "Expected_Entry_Price": f"{entry.price:.4f}" if entry else "",
            "Entry_Time_ET": entry.time_et.isoformat() if entry and entry.time_et else "",
            "Stop_Price": (
                f"{entry.stop_price:.4f}"
                if entry and entry.stop_price is not None else ""
            ),
            "Session_Low": f"{session_low:.4f}" if session_low is not None else "",
            "Stop_Status": stop_status,
            "Stop_Checked_Through_ET": (
                stop_checked_through.isoformat() if stop_checked_through else ""
            ),
        })
    logger.info(f"Trade log updated → {TRADE_LOG_PATH}")


def _print_stop_panel(check: StopCheck) -> None:
    print("\n  RISK / 6% STOP")
    if check.entry:
        print(f"  Expected entry   : ${check.entry.price:.2f}")
        if check.entry.stop_price is not None:
            print(f"  Stop price       : ${check.entry.stop_price:.2f}")
    else:
        print("  Expected entry   : unavailable (legacy/manual position)")

    if check.snapshot:
        print(f"  Price at scan    : ${check.snapshot.last_price:.2f}")
        print(f"  Session low      : ${check.snapshot.session_low:.2f}")

    labels = {
        "ACTIVE": "ACTIVE — not touched since last check",
        "TRIGGERED": "TRIGGERED — model position moves to CASH",
        "NO_ENTRY_PRICE": "UNAVAILABLE — waiting for the next recorded SWITCH",
        "NO_SESSION_DATA": "UNAVAILABLE — no new regular-session minute data",
        "DISABLED": "DISABLED — STOP_LOSS_PCT is None",
        "CASH": "N/A — model position is CASH",
    }
    print(f"  Stop status      : {labels.get(check.status, check.status)}")


def _print_decision_panel(action: str, reason: str) -> None:
    width = 57
    title = f"DECISION :  *** {action} ***"
    print()
    print(f"  ┌{'─' * (width + 2)}┐")
    print(f"  │ {title:<{width}} │")
    for line in textwrap.wrap(reason, width=width) or [""]:
        print(f"  │ {line:<{width}} │")
    print(f"  └{'─' * (width + 2)}┘")


# ─────────────────────────────────────────────────────────────────────────────
# Hold discipline (mirrors the backtest's MIN_HOLD_DAYS + SWITCH_CONFIRM_DAYS)
# ─────────────────────────────────────────────────────────────────────────────

def scans_held(current: str) -> int:
    """
    How many prior scan dates (from trade_log.csv) the current ticker has
    already been the holding, counting back from the most recent entries.
    Today's own rows are ignored. Returns 0 if unknown/new.
    """
    if current == "CASH" or not os.path.exists(TRADE_LOG_PATH):
        return 0
    try:
        with open(TRADE_LOG_PATH, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return 0

    today = date.today().isoformat()
    dates: list[str] = []
    for r in reversed(rows):
        d = r.get("Date", "")
        if d == today:
            if r.get("Current_Position") != current:
                break
            continue  # today doesn't count as a day held
        if r.get("Current_Position") != current:
            break
        if not dates or dates[-1] != d:
            dates.append(d)
    return len(dates)


def _load_state() -> dict:
    try:
        with open(SCANNER_STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        with open(SCANNER_STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save scanner state: {e}")


def update_edge_streak(has_edge: bool) -> int:
    """
    Track how many CONSECUTIVE scan days the switch edge has persisted.
    A day without an edge (or without a scan) resets the streak.
    Multiple scans on the same day count once.
    """
    state     = _load_state()
    today     = date.today().isoformat()
    prev_scan = state.get("last_scan_date")
    prev_edge = state.get("last_edge_date")
    streak    = int(state.get("edge_streak", 0))

    if has_edge:
        if prev_edge == today:
            pass                                   # already counted today
        elif prev_scan is not None and prev_scan == prev_edge:
            streak += 1                            # previous scan day also had it
        else:
            streak = 1
        state["last_edge_date"] = today
    else:
        streak = 0
        state.pop("last_edge_date", None)

    state["last_scan_date"] = today
    state["edge_streak"]    = streak
    _save_state(state)
    return streak


def reset_edge_streak() -> None:
    state = _load_state()
    state["edge_streak"] = 0
    state.pop("last_edge_date", None)
    state["last_scan_date"] = date.today().isoformat()
    _save_state(state)


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for score visualisation."""
    if pd.isna(score):
        return f"[{'░' * width}]   n/a"
    filled = round(score * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {score:.3f}"


def _print_header():
    now = datetime.now().strftime("%A, %d %B %Y  %H:%M")
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          DAILY LONG-ONLY ROTATION SCANNER                ║")
    print(f"║  {now:<56}║")
    print("╚══════════════════════════════════════════════════════════╝")


def _print_table(df: pd.DataFrame, current_ticker: str, n: int):
    print(f"\n  TOP {n} CANDIDATES\n  {'─'*54}")
    header = f"  {'Ticker':<8} {'Score':>6}  {'Momentum':>8} {'Volume':>6} {'Event':>5} {'Flow':>5}  Bar"
    print(header)
    print(f"  {'─'*54}")
    for _, row in df.head(n).iterrows():
        marker = " ◄ CURRENT" if row["Ticker"] == current_ticker else ""
        bar    = _bar(row["Score"])
        print(
            f"  {row['Ticker']:<8} {row['Score']:>6.3f}  "
            f"{row['Momentum']:>8.3f} {row['Volume']:>6.3f} "
            f"{row['Event']:>5.3f} {row['Flow']:>5.3f}  "
            f"{bar}{marker}"
        )
    print(f"  {'─'*54}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _print_header()

    # ── 1. Current position ───────────────────────────────────────────────────
    current_ticker = get_current_ticker()
    print(f"\n  Current holding : {current_ticker}")

    # ── Live model stop check ────────────────────────────────────────────────
    # The saved price is explicitly an estimate taken when Relay emitted the
    # SWITCH. Broker fills remain the source of truth for an actual stop order.
    current_entry = find_entry_details(TRADE_LOG_PATH, current_ticker)
    current_snapshot = None
    if current_entry is not None and STOP_LOSS_PCT is not None:
        try:
            current_snapshot = fetch_regular_session_snapshot(
                current_ticker,
                since_et=(
                    current_entry.checked_through_et or current_entry.time_et
                ),
            )
        except Exception as ex:
            logger.warning(f"Intraday stop data unavailable for {current_ticker}: {ex}")
    stop_check = evaluate_stop(current_ticker, current_entry, current_snapshot)
    _print_stop_panel(stop_check)

    if stop_check.triggered:
        reason = (
            f"{current_ticker} session low ${stop_check.snapshot.session_low:.2f} "
            f"touched the model stop ${current_entry.stop_price:.2f} "
            f"from expected entry ${current_entry.price:.2f}; move model to CASH"
        )
        reset_edge_streak()
        _print_decision_panel("STOP_LOSS", reason)
        print("\n  → Model position after stop : CASH")
        append_trade_log(
            current_ticker,
            "STOP_LOSS",
            "CASH",
            reason,
            entry=current_entry,
            session_low=stop_check.snapshot.session_low,
            stop_status=stop_check.status,
            stop_checked_through=stop_check.snapshot.last_time_et,
        )
        print(f"\n  Log saved → {TRADE_LOG_PATH}")
        print()
        return

    # ── 2. Market context ─────────────────────────────────────────────────────
    spy_trend = get_spy_trend()
    direction = "▲" if spy_trend >= 0 else "▼"
    print(f"  SPY 5-day trend : {direction} {spy_trend:+.2%}")

    # ── Regime check ──────────────────────────────────────────────────────────
    from signals import regime_is_bullish
    try:
        spy_hist = yf.download("SPY", period="80d", progress=False, auto_adjust=True)
        bull = regime_is_bullish(spy_hist)
    except Exception:
        bull = True  # default to bullish if check fails

    regime_str = "✅ BULL — long positions permitted" if bull else "⚠️  BEAR — regime filter active, consider CASH"
    print(f"  Regime          : {regime_str}")

    if not bull:
        print("\n  ┌─────────────────────────────────────────────────────┐")
        print("  │  REGIME FILTER: SPY below 50-day SMA               │")
        print("  │  Recommendation: Stay in CASH until regime clears  │")
        print("  └─────────────────────────────────────────────────────┘")
        print()

    # ── 3. Universe ───────────────────────────────────────────────────────────
    universe = get_universe()
    if not universe:
        logger.error("Empty universe — check internet connection and Wikipedia sources.")
        return

    # Always include current holding in the scan
    if current_ticker not in universe and current_ticker != "CASH":
        universe.insert(0, current_ticker)

    # ── 4. Bulk OHLCV + liquidity filter ──────────────────────────────────────
    data = bulk_download(universe)
    data = apply_liquidity_filter(data)

    if not data:
        logger.error("No data survived the liquidity filter.")
        return

    # ── 5. Price/volume scores for ALL tickers → keep top PRE_FILTER_TOP_N ────
    # Momentum is ranked cross-sectionally against the full filtered universe
    # (not just the enriched subset) so percentiles mean the same thing the
    # backtest validated.
    m_scores, v_scores = price_volume_scores(data)
    price_weight = WEIGHTS["momentum"] + WEIGHTS["volume"]
    qs = {
        t: (
            WEIGHTS["momentum"] * m_scores[t]
            + WEIGHTS["volume"] * v_scores[t]
        ) / price_weight
        for t in data
    }
    top = sorted(qs, key=qs.get, reverse=True)[:PRE_FILTER_TOP_N]

    # Ensure current position is always evaluated
    if current_ticker in data and current_ticker not in top:
        top.append(current_ticker)
        logger.info(f"Forced {current_ticker} into full-score set")

    # ── 6. Full scoring (event + flow calls per ticker) ───────────────────────
    n = len(top)
    logger.info(f"Full scoring {n} candidates…")
    print(f"\n  Enriching top {n} candidates (this takes ~{n // 10 + 1} min)…")

    records = []
    reset_flow_stats()
    for i, t in enumerate(top, 1):
        logger.info(f"  [{i:>3}/{n}]  {t}")
        try:
            records.append(full_score(t, m_scores[t], v_scores[t], spy_trend))
        except Exception as ex:
            logger.warning(f"  Skipped {t}: {ex}")

    fs = flow_stats()
    n_missing = fs["attempts"] - fs["gap_ok"]
    if fs["attempts"] and n_missing:
        logger.warning(
            f"Flow gap data unavailable for {n_missing}/{fs['attempts']} tickers "
            f"(Yahoo empty/throttled responses) — those used market context only"
        )

    if not records:
        logger.error("All full-score attempts failed.")
        return

    # ── 7. Sort results ───────────────────────────────────────────────────────
    df_results = (
        pd.DataFrame(records)
        .dropna(subset=["Score"])
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )
    if df_results.empty:
        logger.error("All scores were NaN — likely bad/partial price data.")
        return

    # ── 8. Score of current position ─────────────────────────────────────────
    curr_rows = df_results[df_results["Ticker"] == current_ticker]
    if not curr_rows.empty:
        current_score = float(curr_rows.iloc[0]["Score"])
    else:
        current_score = 0.50   # neutral (CASH or no data)
        logger.info(f"{current_ticker} not scored — using neutral 0.50")

    # ── 9. Best candidate (excluding current) ────────────────────────────────
    candidates = df_results[df_results["Ticker"] != current_ticker]
    if candidates.empty:
        print("\n  No alternative candidates found — HOLD by default.")
        return

    best_ticker = candidates.iloc[0]["Ticker"]
    best_score  = float(candidates.iloc[0]["Score"])

    # ── 10. Display ───────────────────────────────────────────────────────────
    # Make sure current position appears in the displayed table if it's ranked lower
    display_df = df_results.copy()
    if current_ticker not in display_df.head(TOP_CANDIDATES)["Ticker"].values:
        if not curr_rows.empty:
            display_df = pd.concat(
                [display_df.head(TOP_CANDIDATES), curr_rows]
            ).drop_duplicates("Ticker")

    _print_table(display_df, current_ticker, TOP_CANDIDATES)

    # Current position panel
    print(f"\n  CURRENT POSITION : {current_ticker}")
    print(f"  Score            : {_bar(current_score)}")

    # ── 11. Decision ──────────────────────────────────────────────────────────
    action, reason = decision(current_ticker, current_score, best_ticker, best_score)

    # Min-hold discipline (mirrors backtest MIN_HOLD_DAYS; CASH is exempt)
    if action == "SWITCH" and current_ticker != "CASH":
        held = scans_held(current_ticker)
        if held < MIN_HOLD_DAYS:
            action = "HOLD"
            reason = (
                f"Edge to {best_ticker} exists, but {current_ticker} held only "
                f"{held}/{MIN_HOLD_DAYS} scan days — min-hold enforced"
            )
            reset_edge_streak()

    # Switch confirmation: edge must persist SWITCH_CONFIRM_DAYS consecutive
    # scan days before firing (mirrors backtest; CASH entries are immediate)
    if current_ticker != "CASH" and SWITCH_CONFIRM_DAYS > 1:
        if action == "SWITCH":
            streak = update_edge_streak(True)
            if streak < SWITCH_CONFIRM_DAYS:
                action = "HOLD"
                reason = (
                    f"Edge to {best_ticker} (Δ={best_score - current_score:+.3f}) "
                    f"on scan day {streak}/{SWITCH_CONFIRM_DAYS} — awaiting confirmation"
                )
            else:
                reset_edge_streak()
        else:
            update_edge_streak(False)

    _print_decision_panel(action, reason)

    if action == "SWITCH":
        print(f"\n  → Consider switching to : {best_ticker}  (score {best_score:.3f})")

    # ── 12. Log ───────────────────────────────────────────────────────────────
    log_entry = current_entry
    log_session_low = (
        current_snapshot.session_low if current_snapshot is not None else None
    )
    log_stop_status = stop_check.status

    if action == "SWITCH":
        new_snapshot = None
        try:
            new_snapshot = fetch_regular_session_snapshot(best_ticker)
        except Exception as ex:
            logger.warning(f"Could not estimate entry for {best_ticker}: {ex}")

        estimated_price = (
            new_snapshot.last_price
            if new_snapshot is not None
            else float(data[best_ticker]["Close"].dropna().iloc[-1])
        )
        entry_time = (
            new_snapshot.last_time_et
            if new_snapshot is not None
            else datetime.now(ZoneInfo("America/New_York"))
        )
        log_entry = EntryDetails(
            ticker=best_ticker,
            price=estimated_price,
            time_et=entry_time,
            stop_price=(
                estimated_price * (1 + STOP_LOSS_PCT)
                if STOP_LOSS_PCT is not None else None
            ),
            checked_through_et=entry_time,
        )
        log_session_low = None
        log_stop_status = "ENTRY_ESTIMATE"
        print(f"  Expected entry   : ${log_entry.price:.2f}")
        if log_entry.stop_price is not None:
            print(f"  6% model stop    : ${log_entry.stop_price:.2f}")

    append_trade_log(
        current_ticker,
        action,
        best_ticker,
        reason,
        entry=log_entry,
        session_low=log_session_low,
        stop_status=log_stop_status,
        stop_checked_through=(
            log_entry.checked_through_et
            if action == "SWITCH" and log_entry is not None
            else current_snapshot.last_time_et if current_snapshot is not None
            else current_entry.checked_through_et if current_entry is not None
            else None
        ),
    )

    print(f"\n  Log saved → {TRADE_LOG_PATH}")
    print()


if __name__ == "__main__":
    main()
