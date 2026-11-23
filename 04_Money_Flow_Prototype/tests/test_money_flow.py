import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hkjc_money_flow import (  # noqa: E402
    MoneyFlowProcessor,
    OutputStore,
    OutputWriter,
    format_utc,
    load_offline_records,
    parse_timestamp,
)


def snapshot(
    observed="2026-01-05T02:00:00Z",
    investment=10_000_000,
    odds=None,
    record_id="TEST-001",
    odds_time=None,
    pool_time=None,
    race_no=1,
):
    return {
        "record_id": record_id,
        "source": "SYNTHETIC",
        "is_synthetic": True,
        "observed_at_utc": observed,
        "meeting_date": "2026-01-05",
        "venue": "ST",
        "race_no": race_no,
        "pool_id": f"SYNTH-20260105-ST-R{race_no}-WIN",
        "status": "OPEN" if odds else "DEFINED",
        "sell_status": "SELLING" if odds else "STOP_SELL",
        "odds_source_time": observed if odds_time is None else odds_time,
        "pool_source_time": observed if pool_time is None else pool_time,
        "pool_investment": investment,
        "odds": [{"combString": "1", "oddsValue": 5.0}] if odds is None else odds,
    }


class MoneyFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = OutputStore(self.temp.name)
        self.processor = MoneyFlowProcessor(self.store, stale_after_seconds=0)

    def tearDown(self):
        self.temp.cleanup()

    def runner(self, rows, number="1"):
        return next(row for row in rows if row["runner_no"] == number)

    def test_ten_million_example(self):
        _, rows = self.processor.process(snapshot())
        row = self.runner(rows)
        self.assertAlmostEqual(float(row["implied_stake"]), 1_650_000.0, places=6)
        self.assertAlmostEqual(float(row["implied_stake_share"]), 0.165, places=12)
        self.assertEqual(row["flow_status"], "first_valid_snapshot")

    def test_interval_flow_example(self):
        self.processor.process(snapshot())
        second = snapshot(
            observed="2026-01-05T02:05:00Z",
            investment=11_000_000,
            odds=[{"combString": "1", "oddsValue": 4.5}],
            record_id="TEST-002",
        )
        _, rows = self.processor.process(second)
        row = self.runner(rows)
        self.assertAlmostEqual(float(row["implied_stake"]), 2_016_666.6666666667, places=5)
        self.assertAlmostEqual(float(row["implied_stake_flow"]), 366_666.6666666667, places=5)
        self.assertAlmostEqual(float(row["interval_seconds"]), 300.0)
        self.assertEqual(row["flow_status"], "estimated")

    def test_pool_growth_with_unchanged_odds_has_zero_share_change(self):
        self.processor.process(snapshot())
        second = snapshot(
            observed="2026-01-05T02:05:00Z",
            investment=11_000_000,
            record_id="TEST-002",
        )
        _, rows = self.processor.process(second)
        row = self.runner(rows)
        self.assertAlmostEqual(float(row["implied_stake_flow"]), 165_000.0, places=6)
        self.assertAlmostEqual(float(row["implied_stake_share_change"]), 0.0, places=12)
        self.assertEqual(row["odds_changed"], 0)

    def test_negative_implied_flow_is_retained_and_flagged(self):
        self.processor.process(snapshot())
        second = snapshot(
            observed="2026-01-05T02:05:00Z",
            odds=[{"combString": "1", "oddsValue": 10.0}],
            record_id="TEST-002",
        )
        snap, rows = self.processor.process(second)
        row = self.runner(rows)
        self.assertLess(float(row["implied_stake_flow"]), 0)
        self.assertEqual(row["is_negative_flow"], 1)
        self.assertIn("NEGATIVE_IMPLIED_FLOW", row["quality_flags"])
        self.assertIn("NEGATIVE_IMPLIED_FLOW", snap["quality_flags"])

    def test_missing_investment_suppresses_stake_calculation(self):
        snap, rows = self.processor.process(snapshot(investment=None))
        self.assertEqual(snap["calculation_status"], "not_calculated")
        self.assertIn("MISSING_INVESTMENT", snap["quality_flags"])
        self.assertEqual(rows[0]["implied_stake"], "")
        self.assertEqual(rows[0]["flow_status"], "not_calculated")

    def test_empty_presale_market_is_not_fabricated(self):
        snap, rows = self.processor.process(snapshot(investment=None, odds=[]))
        self.assertEqual(snap["calculation_status"], "no_market_data")
        self.assertEqual(rows, [])
        self.assertIn("EMPTY_ODDS_NODES", snap["quality_flags"])
        self.assertIn("NOT_SELLING", snap["quality_flags"])

    def test_invalid_odds_are_preserved_but_not_calculated(self):
        snap, rows = self.processor.process(
            snapshot(odds=[{"combString": "7", "oddsValue": "SCR"}])
        )
        self.assertEqual(rows[0]["raw_odds_value"], "SCR")
        self.assertEqual(rows[0]["implied_stake"], "")
        self.assertIn("INVALID_ODDS", rows[0]["quality_flags"])
        self.assertIn("PARTIAL_ODDS_INVALID", snap["quality_flags"])

    def test_source_timestamp_mismatch(self):
        snap, _ = self.processor.process(
            snapshot(pool_time="2026-01-05T01:50:00Z")
        )
        self.assertIn("SOURCE_TIMESTAMP_MISMATCH", snap["quality_flags"])
        self.assertAlmostEqual(snap["source_gap_seconds"], 600.0)

    def test_stale_source_timestamp(self):
        processor = MoneyFlowProcessor(self.store, stale_after_seconds=300)
        snap, _ = processor.process(
            snapshot(
                observed="2026-01-05T02:00:00Z",
                odds_time="2026-01-05T01:50:00Z",
                pool_time="2026-01-05T01:50:00Z",
            )
        )
        self.assertIn("STALE_ODDS_SOURCE", snap["quality_flags"])
        self.assertIn("STALE_POOL_SOURCE", snap["quality_flags"])

    def test_unchanged_derived_state_is_deduplicated(self):
        first = snapshot()
        self.processor.process(first)
        second = snapshot(
            observed="2026-01-05T02:05:00Z",
            odds_time="2026-01-05T02:05:00Z",
            pool_time="2026-01-05T02:05:00Z",
            record_id="TEST-002",
        )
        snap, rows = self.processor.process(second)
        self.assertEqual(snap["calculation_status"], "duplicate_unchanged_state")
        self.assertEqual(snap["is_duplicate"], 1)
        self.assertEqual(rows, [])

    def test_pool_closure_for_coherent_market(self):
        target_inverse_sum = 1 / 0.825
        d3 = 1 / (target_inverse_sum - 1 / 5 - 1 / 3)
        odds = [
            {"combString": "1", "oddsValue": 5.0},
            {"combString": "2", "oddsValue": 3.0},
            {"combString": "3", "oddsValue": d3},
        ]
        snap, _ = self.processor.process(snapshot(odds=odds))
        self.assertAlmostEqual(snap["sum_implied_stake"], 10_000_000.0, places=5)
        self.assertAlmostEqual(snap["pool_closure_relative"], 0.0, places=12)
        self.assertNotIn("POOL_CLOSURE_LARGE", snap["quality_flags"])

    def test_raw_graphql_structure_and_meeting_id(self):
        record = snapshot()
        record.update(
            {
                "record_id": "RAW-001",
                "request": {"venueCode": "ST", "raceNo": 1, "date": None},
                "odds_response": {
                    "data": {
                        "raceMeetings": [
                            {
                                "pmPools": [
                                    {
                                        "id": "MTG_20260105_0001WIN1",
                                        "status": "OPEN",
                                        "sellStatus": "SELLING",
                                        "oddsType": "WIN",
                                        "lastUpdateTime": "2026-01-05T02:00:00Z",
                                        "leg": {"races": [1]},
                                        "oddsNodes": [{"combString": "01", "oddsValue": "5.0"}],
                                    }
                                ]
                            }
                        ]
                    }
                },
                "pool_response": {
                    "data": {
                        "raceMeetings": [
                            {
                                "poolInvs": [
                                    {
                                        "id": "MTG_20260105_0001WIN1",
                                        "status": "OPEN",
                                        "sellStatus": "SELLING",
                                        "oddsType": "WIN",
                                        "investment": "10,000,000",
                                        "lastUpdateTime": "2026-01-05T02:00:00Z",
                                        "leg": {"races": [1]},
                                    }
                                ]
                            }
                        ]
                    }
                },
            }
        )
        # Raw format takes precedence over the simplified helper keys.
        snap, rows = self.processor.process(record)
        self.assertEqual(snap["meeting_date"], "2026-01-05")
        self.assertEqual(rows[0]["runner_no"], "1")
        self.assertAlmostEqual(float(rows[0]["implied_stake"]), 1_650_000.0)

        record["record_id"] = "RAW-002"
        record["pool_response"]["data"]["raceMeetings"][0]["poolInvs"][0]["id"] = (
            "MTG_20260105_0001WIN9"
        )
        mismatch_snap, _ = self.processor.process(record)
        self.assertIn("POOL_ID_MISMATCH", mismatch_snap["quality_flags"])

    def test_restart_loads_previous_valid_state(self):
        writer = OutputWriter(self.store, self.processor, quiet=True)
        writer.handle(snapshot())
        restarted_processor = MoneyFlowProcessor(self.store, stale_after_seconds=0)
        restarted_writer = OutputWriter(self.store, restarted_processor, quiet=True)
        restarted_writer.handle(
            snapshot(
                observed="2026-01-05T02:05:00Z",
                investment=11_000_000,
                odds=[{"combString": "1", "oddsValue": 4.5}],
                record_id="TEST-002",
            )
        )
        import csv
        with self.store.runners_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[1]["implied_stake_flow"]), 366_666.6666666667, places=5)

    def test_naive_source_timestamp_is_hong_kong_time(self):
        parsed = parse_timestamp("2026-01-05 10:00:00")
        self.assertEqual(format_utc(parsed), "2026-01-05T02:00:00.000Z")

    def test_included_fixture_is_explicitly_synthetic(self):
        records = load_offline_records(ROOT / "examples" / "synthetic_snapshots.json")
        self.assertEqual(len(records), 6)
        self.assertTrue(all(record["is_synthetic"] for record in records))


if __name__ == "__main__":
    unittest.main()
