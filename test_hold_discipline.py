import csv
import json
import os
import tempfile
import unittest
from datetime import date
from unittest.mock import patch

import main as scanner


class HoldDisciplineTest(unittest.TestCase):
    def _write_log(self, path: str, entry_date: str, ticker: str = "MDT") -> None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["Date", "Current_Position", "Action", "Switch_To"],
            )
            writer.writeheader()
            writer.writerow({
                "Date": entry_date,
                "Current_Position": "CRM",
                "Action": "SWITCH",
                "Switch_To": ticker,
            })

    def test_missed_scan_still_counts_completed_nyse_session(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = os.path.join(directory, "trade_log.csv")
            self._write_log(log_path, "2026-09-28")  # Monday

            with patch.object(scanner, "TRADE_LOG_PATH", log_path):
                held = scanner.trading_days_held("MDT", date(2026, 9, 30))

            self.assertEqual(1, held)  # Tuesday counts even without a log row

    def test_weekend_does_not_count_as_hold_day(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = os.path.join(directory, "trade_log.csv")
            self._write_log(log_path, "2026-10-02")  # Friday

            with patch.object(scanner, "TRADE_LOG_PATH", log_path):
                held = scanner.trading_days_held("MDT", date(2026, 10, 5))

            self.assertEqual(0, held)

    def test_nyse_holiday_does_not_count_as_hold_day(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = os.path.join(directory, "trade_log.csv")
            self._write_log(log_path, "2026-07-02")  # Thursday

            with patch.object(scanner, "TRADE_LOG_PATH", log_path):
                held = scanner.trading_days_held("MDT", date(2026, 7, 6))

            # July 3 is the observed Independence Day holiday; the weekend is
            # also excluded, so no completed NYSE session has elapsed.
            self.assertEqual(0, held)

    def test_edge_streak_resets_after_a_missed_nyse_session(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = os.path.join(directory, "scanner_state.json")
            with open(state_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "last_scan_date": "2026-09-28",
                    "last_edge_date": "2026-09-28",
                    "edge_streak": 1,
                }, handle)

            with patch.object(scanner, "SCANNER_STATE_PATH", state_path):
                streak = scanner.update_edge_streak(True, date(2026, 9, 30))

            self.assertEqual(1, streak)

    def test_edge_streak_continues_across_weekend(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = os.path.join(directory, "scanner_state.json")
            with open(state_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "last_scan_date": "2026-10-02",
                    "last_edge_date": "2026-10-02",
                    "edge_streak": 1,
                }, handle)

            with patch.object(scanner, "SCANNER_STATE_PATH", state_path):
                streak = scanner.update_edge_streak(True, date(2026, 10, 5))

            self.assertEqual(2, streak)


if __name__ == "__main__":
    unittest.main()
