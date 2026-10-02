import csv
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

import main as scanner

from position_risk import (
    EntryDetails,
    MarketSnapshot,
    TRADE_LOG_FIELDS,
    calculate_atr_pct,
    calculate_shadow_stop_price,
    ensure_trade_log_schema,
    evaluate_stop,
    evaluate_shadow_stop,
    find_entry_details,
)


ET = ZoneInfo("America/New_York")


class PositionRiskTest(unittest.TestCase):
    def test_old_trade_log_is_migrated_without_losing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "trade_log.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "Date", "Current_Position", "Action",
                    "Switch_To", "Reason", "Outcome",
                ])
                writer.writerow([
                    "2026-09-01", "CRM", "SWITCH", "MDT", "test", "",
                ])

            ensure_trade_log_schema(path)

            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(TRADE_LOG_FIELDS, reader.fieldnames)
            self.assertEqual(1, len(rows))
            self.assertEqual("MDT", rows[0]["Switch_To"])
            self.assertEqual("", rows[0]["Expected_Entry_Price"])

    def test_latest_switch_entry_is_found(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "trade_log.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=TRADE_LOG_FIELDS)
                writer.writeheader()
                writer.writerow({
                    "Date": "2026-09-01",
                    "Current_Position": "CRM",
                    "Action": "SWITCH",
                    "Switch_To": "MDT",
                    "Expected_Entry_Price": "100",
                    "Entry_Time_ET": "2026-09-01T15:30:00-04:00",
                    "Stop_Price": "94",
                    "ATR14_Pct": "0.04",
                    "Shadow_Stop_Price": "92",
                })

            entry = find_entry_details(path, "MDT")

            self.assertIsNotNone(entry)
            self.assertEqual(100.0, entry.price)
            self.assertEqual(94.0, entry.stop_price)
            self.assertEqual(0.04, entry.atr14_pct)
            self.assertEqual(92.0, entry.shadow_stop_price)
            self.assertEqual(15, entry.time_et.hour)
            self.assertEqual(30, entry.time_et.minute)

    def test_stale_entry_is_not_reused_after_switching_away(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "trade_log.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=TRADE_LOG_FIELDS)
                writer.writeheader()
                writer.writerow({
                    "Date": "2026-09-01", "Current_Position": "CASH",
                    "Action": "SWITCH", "Switch_To": "MDT",
                    "Expected_Entry_Price": "100", "Stop_Price": "94",
                })
                writer.writerow({
                    "Date": "2026-09-02", "Current_Position": "MDT",
                    "Action": "SWITCH", "Switch_To": "NVDA",
                    "Expected_Entry_Price": "125", "Stop_Price": "117.5",
                })

            self.assertIsNone(find_entry_details(path, "MDT"))

    def test_stop_triggers_when_session_low_touches_threshold(self):
        yesterday = datetime.now(ET) - timedelta(days=1)
        now = datetime.now(ET)
        entry = EntryDetails("MDT", 100.0, yesterday, 94.0)
        snapshot = MarketSnapshot("MDT", 96.0, 93.99, now)

        result = evaluate_stop("MDT", entry, snapshot)

        self.assertTrue(result.triggered)
        self.assertEqual("TRIGGERED", result.status)

    def test_atr_and_shadow_stop_use_the_wider_distance(self):
        history = pd.DataFrame({
            "High": [103.0] * 15,
            "Low": [97.0] * 15,
            "Close": [100.0] * 15,
        })

        atr_pct = calculate_atr_pct(history, period=14)
        shadow = calculate_shadow_stop_price(100.0, -0.06, atr_pct, 2.0)

        self.assertAlmostEqual(0.06, atr_pct)
        self.assertAlmostEqual(88.0, shadow)

    def test_shadow_breach_never_changes_hard_stop_result(self):
        now = datetime.now(ET)
        entry = EntryDetails(
            "MDT",
            100.0,
            now,
            94.0,
            atr14_pct=0.04,
            shadow_stop_price=92.0,
        )
        snapshot = MarketSnapshot("MDT", 93.5, 93.0, now)

        hard_result = evaluate_stop("MDT", entry, snapshot)
        shadow_status = evaluate_shadow_stop(entry, snapshot)

        self.assertTrue(hard_result.triggered)
        self.assertEqual("ACTIVE", shadow_status)

    def test_shadow_stop_is_unavailable_without_atr(self):
        self.assertIsNone(
            calculate_shadow_stop_price(100.0, -0.06, None, 2.0)
        )

    def test_trade_log_records_shadow_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "trade_log.csv")
            original_path = scanner.TRADE_LOG_PATH
            scanner.TRADE_LOG_PATH = path
            try:
                entry = EntryDetails(
                    "MDT",
                    100.0,
                    datetime.now(ET),
                    94.0,
                    atr14_pct=0.04,
                    shadow_stop_price=92.0,
                )
                scanner.append_trade_log(
                    "CRM",
                    "SWITCH",
                    "MDT",
                    "test",
                    entry=entry,
                    stop_status="ENTRY_ESTIMATE",
                    shadow_stop_status="ENTRY_ESTIMATE",
                )
            finally:
                scanner.TRADE_LOG_PATH = original_path

            with open(path, newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual("0.040000", row["ATR14_Pct"])
            self.assertEqual("92.0000", row["Shadow_Stop_Price"])
            self.assertEqual("ENTRY_ESTIMATE", row["Shadow_Stop_Status"])

    def test_post_entry_bar_on_entry_day_is_monitored(self):
        now = datetime.now(ET)
        entry = EntryDetails("MDT", 100.0, now, 94.0)
        snapshot = MarketSnapshot("MDT", 96.0, 90.0, now)

        result = evaluate_stop("MDT", entry, snapshot)

        self.assertTrue(result.triggered)
        self.assertEqual("TRIGGERED", result.status)


if __name__ == "__main__":
    unittest.main()
