"""Live model-position entry tracking and intraday stop monitoring."""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


NEW_YORK = ZoneInfo("America/New_York")

TRADE_LOG_FIELDS = [
    "Date",
    "Current_Position",
    "Action",
    "Switch_To",
    "Reason",
    "Outcome",
    "Expected_Entry_Price",
    "Entry_Time_ET",
    "Stop_Price",
    "Session_Low",
    "Stop_Status",
    "Stop_Checked_Through_ET",
]


@dataclass(frozen=True)
class EntryDetails:
    ticker: str
    price: float
    time_et: Optional[datetime]
    stop_price: Optional[float]
    checked_through_et: Optional[datetime] = None


@dataclass(frozen=True)
class MarketSnapshot:
    ticker: str
    last_price: float
    session_low: float
    last_time_et: datetime


@dataclass(frozen=True)
class StopCheck:
    status: str
    entry: Optional[EntryDetails] = None
    snapshot: Optional[MarketSnapshot] = None
    triggered: bool = False


def _float_or_none(value) -> Optional[float]:
    try:
        number = float(value)
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None


def _datetime_or_none(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=NEW_YORK)
        return parsed.astimezone(NEW_YORK)
    except (TypeError, ValueError):
        return None


def read_trade_log(path: str) -> tuple[list[str], list[dict]]:
    if not os.path.exists(path):
        return [], []
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def ensure_trade_log_schema(path: str) -> None:
    """Add risk columns to an old log without discarding existing rows."""
    fieldnames, rows = read_trade_log(path)
    if not fieldnames or all(name in fieldnames for name in TRADE_LOG_FIELDS):
        return

    target_fields = fieldnames + [
        name for name in TRADE_LOG_FIELDS if name not in fieldnames
    ]

    directory = os.path.dirname(os.path.abspath(path))
    fd, temp_path = tempfile.mkstemp(prefix="relay-log-", suffix=".csv", dir=directory)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=target_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in target_fields})
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def find_entry_details(path: str, current_ticker: str) -> Optional[EntryDetails]:
    """Return the latest recorded model entry for the current ticker."""
    ticker = current_ticker.strip().upper()
    if not ticker or ticker == "CASH":
        return None

    _, rows = read_trade_log(path)
    if not rows:
        return None

    last = rows[-1]
    last_action = (last.get("Action") or "").strip().upper()
    last_current = (last.get("Current_Position") or "").strip().upper()
    last_switch_to = (last.get("Switch_To") or "").strip().upper()
    if last_action == "STOP_LOSS":
        latest_model_position = "CASH"
    elif last_action == "SWITCH" and last_switch_to:
        latest_model_position = last_switch_to
    else:
        latest_model_position = last_current
    if latest_model_position != ticker:
        # A manual ticker that disagrees with the model log must not inherit a
        # stale cost basis from an older holding cycle.
        return None

    for row in reversed(rows):
        action = (row.get("Action") or "").strip().upper()
        current = (row.get("Current_Position") or "").strip().upper()
        switch_to = (row.get("Switch_To") or "").strip().upper()

        belongs_to_position = current == ticker or (
            action == "SWITCH" and switch_to == ticker
        )
        if not belongs_to_position:
            continue

        if action == "STOP_LOSS" or (action == "SWITCH" and switch_to != ticker):
            return None

        price = _float_or_none(row.get("Expected_Entry_Price"))
        if price is None:
            continue
        stop_price = _float_or_none(row.get("Stop_Price"))
        return EntryDetails(
            ticker=ticker,
            price=price,
            time_et=_datetime_or_none(row.get("Entry_Time_ET")),
            stop_price=stop_price,
            checked_through_et=_datetime_or_none(
                row.get("Stop_Checked_Through_ET")
            ),
        )
    return None


def fetch_regular_session_snapshot(
    ticker: str,
    since_et: Optional[datetime] = None,
) -> Optional[MarketSnapshot]:
    """Fetch regular-session one-minute prices since the last saved check."""
    history = yf.Ticker(ticker).history(
        period="5d",
        interval="1m",
        prepost=False,
        auto_adjust=False,
    )
    if history.empty or "Close" not in history or "Low" not in history:
        return None

    frame = history.dropna(subset=["Close", "Low"]).copy()
    if frame.empty:
        return None

    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is None:
        index = index.tz_localize(NEW_YORK)
    else:
        index = index.tz_convert(NEW_YORK)
    frame.index = index

    if since_et is not None:
        cutoff = since_et
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=NEW_YORK)
        frame = frame[frame.index > cutoff.astimezone(NEW_YORK)]
    else:
        today_et = datetime.now(NEW_YORK).date()
        frame = frame[frame.index.date == today_et]
    if frame.empty:
        return None

    return MarketSnapshot(
        ticker=ticker,
        last_price=float(frame["Close"].iloc[-1]),
        session_low=float(frame["Low"].min()),
        last_time_et=frame.index[-1].to_pydatetime(),
    )


def evaluate_stop(
    ticker: str,
    entry: Optional[EntryDetails],
    snapshot: Optional[MarketSnapshot],
) -> StopCheck:
    if ticker == "CASH":
        return StopCheck(status="CASH")
    if entry is None:
        return StopCheck(status="NO_ENTRY_PRICE")
    if entry.stop_price is None:
        return StopCheck(status="DISABLED", entry=entry)
    if snapshot is None:
        return StopCheck(status="NO_SESSION_DATA", entry=entry)

    triggered = snapshot.session_low <= entry.stop_price
    return StopCheck(
        status="TRIGGERED" if triggered else "ACTIVE",
        entry=entry,
        snapshot=snapshot,
        triggered=triggered,
    )
