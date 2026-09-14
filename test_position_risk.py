import csv
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from position_risk import (
    EntryDetails,
    MarketSnapshot,
    TRADE_LOG_FIELDS,
    ensure_trade_log_schema,
    evaluate_stop,
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
                })

            entry = find_entry_details(path, "MDT")

            self.assertIsNotNone(entry)
            self.assertEqual(100.0, entry.price)
            self.assertEqual(94.0, entry.stop_price)

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

    def test_post_entry_bar_on_entry_day_is_monitored(self):
        now = datetime.now(ET)
        entry = EntryDetails("MDT", 100.0, now, 94.0)
        snapshot = MarketSnapshot("MDT", 96.0, 90.0, now)

        result = evaluate_stop("MDT", entry, snapshot)

        self.assertTrue(result.triggered)
        self.assertEqual("TRIGGERED", result.status)


if __name__ == "__main__":
    unittest.main()
