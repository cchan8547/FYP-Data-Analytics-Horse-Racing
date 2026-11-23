#!/usr/bin/env python3
"""Timestamped HKJC WIN-odds and implied money-flow prototype.

Read-only data collection. No login, account access, or wagering functionality.
The estimates are market-implied stake allocations, not ticket-level transactions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

try:
    import requests
except ImportError:  # Offline calculation can still run with the standard library.
    requests = None

VERSION = "0.1.0"
ENDPOINT = "https://info.cld.hkjc.com/graphql/base/"
DEFAULT_PAYOUT_RATE = 0.825
HK_TZ = ZoneInfo("Asia/Hong_Kong")

WIN_ODDS_QUERY = """
query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
  raceMeetings(date: $date, venueCode: $venueCode) {
    pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
      id
      status
      sellStatus
      oddsType
      lastUpdateTime
      guarantee
      minTicketCost
      name_en
      name_ch
      leg { number races }
      cWinSelections { composite name_ch name_en starters }
      oddsNodes {
        combString
        oddsValue
        hotFavourite
        oddsDropValue
        bankerOdds { combString oddsValue }
      }
    }
  }
}
"""

WIN_POOL_QUERY = """
query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
  raceMeetings(date: $date, venueCode: $venueCode) {
    totalInvestment
    poolInvs: pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
      id
      leg { number races }
      status
      sellStatus
      oddsType
      investment
      mergedPoolId
      lastUpdateTime
    }
  }
}
"""

SNAPSHOT_FIELDS = [
    "snapshot_id", "record_id", "state_key", "observed_at_utc",
    "odds_observed_at_utc", "pool_observed_at_utc", "source",
    "is_synthetic", "meeting_date", "venue", "race_no", "pool_id",
    "pool_status", "sell_status", "leg_races", "pool_investment",
    "odds_source_time_raw", "pool_source_time_raw",
    "odds_source_time_utc", "pool_source_time_utc",
    "source_gap_seconds", "runner_count", "valid_odds_count",
    "sum_implied_stake", "sum_implied_stake_share", "pool_closure_error",
    "pool_closure_relative", "payout_rate", "odds_divisor",
    "calculation_status", "is_duplicate", "quality_flags", "error_message",
]

RUNNER_FIELDS = [
    "snapshot_id", "record_id", "state_key", "observed_at_utc", "source",
    "is_synthetic", "meeting_date", "venue", "race_no", "pool_id",
    "runner_no", "comb_string", "raw_odds_value", "decimal_odds",
    "hot_favourite", "odds_drop_value", "pool_investment", "payout_rate",
    "implied_stake", "implied_stake_share", "previous_snapshot_id",
    "previous_observed_at_utc", "previous_pool_investment",
    "previous_decimal_odds", "previous_implied_stake",
    "previous_implied_stake_share", "interval_seconds",
    "pool_investment_change", "implied_stake_flow",
    "implied_stake_share_change", "implied_flow_per_minute", "odds_changed",
    "flow_status", "is_negative_flow", "quality_flags",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def format_utc(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def parse_timestamp(value: Any) -> datetime | None:
    """Parse ISO/common timestamps; naive source times are treated as Hong Kong time."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        number = float(value)
        if number > 1e12:
            number /= 1000.0
        return datetime.fromtimestamp(number, tz=timezone.utc)
    else:
        text = str(value).strip()
        if not text:
            return None
        date_match = re.fullmatch(r"/Date\((-?\d+)(?:[+-]\d{4})?\)/", text)
        if date_match:
            return datetime.fromtimestamp(
                int(date_match.group(1)) / 1000.0, tz=timezone.utc
            )
        if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            return parse_timestamp(float(text))
        normalized = text.replace("Z", "+00:00")
        dt = None
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            for pattern in (
                "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d %H:%M:%S",
                "%Y%m%d%H%M%S",
                "%Y-%m-%d %H:%M",
                "%Y/%m/%d %H:%M",
            ):
                try:
                    dt = datetime.strptime(text, pattern)
                    break
                except ValueError:
                    continue
        if dt is None:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=HK_TZ)
    return dt.astimezone(timezone.utc)


def to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        number = float(text)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def normalized_date(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    for pattern in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            pass
    return text


def runner_number(comb_string: Any) -> str:
    text = "" if comb_string is None else str(comb_string).strip()
    if re.fullmatch(r"\d+", text):
        return str(int(text))
    return text


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class HKJCClient:
    """Small read-only client using the two public queries verified for this prototype."""

    def __init__(self, timeout: float = 15.0, retries: int = 3, endpoint: str = ENDPOINT):
        if requests is None:
            raise RuntimeError("Live collection requires: pip install -r requirements.txt")
        self.timeout = timeout
        self.retries = max(1, retries)
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "AcademicCapstoneResearch/0.1 (read-only odds research)",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def _request(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.post(
                    self.endpoint,
                    json={"query": query, "variables": variables},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("GraphQL response is not a JSON object")
                return payload
            except Exception as exc:  # Network/HTTP/JSON errors are retained in raw output.
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"HKJC request failed after {self.retries} attempt(s): {last_error}")

    def fetch_race(self, race_no: int, venue: str, date: str | None = None) -> dict[str, Any]:
        variables = {
            "date": date,
            "venueCode": venue,
            "oddsTypes": ["WIN"],
            "raceNo": race_no,
        }
        record: dict[str, Any] = {
            "record_version": 1,
            "record_id": f"live-{uuid.uuid4()}",
            "source": "HKJC_PUBLIC_GRAPHQL",
            "is_synthetic": False,
            "collection_started_at_utc": utc_now(),
            "request": {
                "endpoint": self.endpoint,
                "date": date,
                "venueCode": venue,
                "raceNo": race_no,
                "oddsTypes": ["WIN"],
            },
        }
        errors = []
        try:
            record["odds_response"] = self._request(WIN_ODDS_QUERY, variables)
        except Exception as exc:
            record["odds_response"] = None
            errors.append(f"odds: {exc}")
        record["odds_observed_at_utc"] = utc_now()
        try:
            record["pool_response"] = self._request(WIN_POOL_QUERY, variables)
        except Exception as exc:
            record["pool_response"] = None
            errors.append(f"pool: {exc}")
        record["pool_observed_at_utc"] = utc_now()
        record["observed_at_utc"] = record["pool_observed_at_utc"]
        if errors:
            record["collection_error"] = " | ".join(errors)
        return record


class OutputStore:
    """Append-only JSONL input audit plus normalized CSV outputs."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.output_dir / "raw_snapshots.jsonl"
        self.snapshots_path = self.output_dir / "race_snapshots.csv"
        self.runners_path = self.output_dir / "runner_money_flow.csv"
        self.events_path = self.output_dir / "collection_events.jsonl"

    def append_raw(self, record: dict[str, Any]) -> None:
        with self.raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def append_event(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in fields})

    def append_snapshot(self, row: dict[str, Any]) -> None:
        self._append_csv(self.snapshots_path, SNAPSHOT_FIELDS, row)

    def append_runner(self, row: dict[str, Any]) -> None:
        self._append_csv(self.runners_path, RUNNER_FIELDS, row)

    def known_state_keys(self) -> set[str]:
        if not self.runners_path.exists() and not self.snapshots_path.exists():
            return set()
        keys: set[str] = set()
        if self.snapshots_path.exists():
            with self.snapshots_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("state_key"):
                        keys.add(row["state_key"])
        return keys


def extract_win_pool(
    response: dict[str, Any] | None, alias: str, requested_race: int | None
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if response is None:
        return None, ["MISSING_API_RESPONSE"]
    if response.get("errors"):
        errors.append("GRAPHQL_ERROR")
    try:
        meetings = response.get("data", {}).get("raceMeetings", []) or []
    except AttributeError:
        return None, errors + ["INVALID_API_STRUCTURE"]
    candidates: list[dict[str, Any]] = []
    for meeting in meetings:
        for pool in meeting.get(alias, []) or []:
            if pool.get("oddsType") not in (None, "WIN"):
                continue
            races = (pool.get("leg") or {}).get("races") or []
            if requested_race is None or not races or requested_race in races:
                candidates.append(pool)
    if not candidates:
        errors.append("NO_WIN_POOL")
        return None, errors
    return candidates[0], errors


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Accept either a raw GraphQL collection record or the documented offline format."""
    request = record.get("request") or {}
    requested_race = record.get("race_no", request.get("raceNo"))
    try:
        requested_race = int(requested_race) if requested_race is not None else None
    except (TypeError, ValueError):
        requested_race = None

    flags: list[str] = []
    error_parts: list[str] = []
    if record.get("collection_error"):
        flags.append("COLLECTION_ERROR")
        error_parts.append(str(record["collection_error"]))

    simplified_input = (
        ("odds" in record or "pool_investment" in record)
        and "odds_response" not in record
        and "pool_response" not in record
    )
    if simplified_input:
        race_for_leg = requested_race if requested_race is not None else record.get("race_no")
        odds_pool = {
            "id": record.get("pool_id", ""),
            "status": record.get("status", ""),
            "sellStatus": record.get("sell_status", record.get("sellStatus", "")),
            "oddsType": "WIN",
            "lastUpdateTime": record.get("odds_source_time", ""),
            "leg": {"races": [race_for_leg] if race_for_leg is not None else []},
            "oddsNodes": record.get("odds", []) or [],
        }
        pool_pool = {
            "id": record.get("pool_id", ""),
            "status": record.get("status", ""),
            "sellStatus": record.get("sell_status", record.get("sellStatus", "")),
            "oddsType": "WIN",
            "investment": record.get("pool_investment"),
            "lastUpdateTime": record.get("pool_source_time", ""),
            "leg": {"races": [race_for_leg] if race_for_leg is not None else []},
        }
    else:
        odds_pool, odds_errors = extract_win_pool(
            record.get("odds_response"), "pmPools", requested_race
        )
        pool_pool, pool_errors = extract_win_pool(
            record.get("pool_response"), "poolInvs", requested_race
        )
        flags.extend(odds_errors)
        flags.extend(pool_errors)
        for response_name in ("odds_response", "pool_response"):
            response = record.get(response_name)
            if isinstance(response, dict) and response.get("errors"):
                error_parts.append(f"{response_name}: {json_text(response['errors'])}")

    odds_pool = odds_pool or {}
    pool_pool = pool_pool or {}
    odds_pool_id = str(odds_pool.get("id") or "")
    investment_pool_id = str(pool_pool.get("id") or "")
    if odds_pool_id and investment_pool_id and odds_pool_id != investment_pool_id:
        flags.append("POOL_ID_MISMATCH")
    pool_id = str(investment_pool_id or odds_pool_id or record.get("pool_id") or "")
    leg_races = (odds_pool.get("leg") or {}).get("races") or (
        pool_pool.get("leg") or {}
    ).get("races") or []
    race_no = requested_race
    if race_no is None and leg_races:
        try:
            race_no = int(leg_races[0])
        except (TypeError, ValueError):
            race_no = leg_races[0]

    meeting_date = normalized_date(record.get("meeting_date") or request.get("date"))
    id_date = re.search(r"MTG_(\d{8})_", pool_id)
    if id_date:
        meeting_date = normalized_date(id_date.group(1))
    venue = str(record.get("venue") or request.get("venueCode") or "UNKNOWN").upper()
    observed_raw = record.get("observed_at_utc")
    observed = parse_timestamp(observed_raw)
    if observed is None:
        flags.append("MISSING_OBSERVED_TIME")
        observed = datetime.now(timezone.utc)
    odds_source_raw = odds_pool.get("lastUpdateTime") or ""
    pool_source_raw = pool_pool.get("lastUpdateTime") or ""
    odds_source = parse_timestamp(odds_source_raw)
    pool_source = parse_timestamp(pool_source_raw)
    if odds_source_raw and odds_source is None:
        flags.append("UNPARSEABLE_ODDS_SOURCE_TIME")
    if pool_source_raw and pool_source is None:
        flags.append("UNPARSEABLE_POOL_SOURCE_TIME")

    investment = to_float(pool_pool.get("investment"))
    return {
        "record_id": str(record.get("record_id") or f"record-{uuid.uuid4()}"),
        "source": str(record.get("source") or "OFFLINE"),
        "is_synthetic": bool(record.get("is_synthetic", False)),
        "odds_convention_verified": bool(record.get("odds_convention_verified", False)),
        "observed": observed,
        "observed_raw": observed_raw or "",
        "odds_observed_at_utc": record.get("odds_observed_at_utc", ""),
        "pool_observed_at_utc": record.get("pool_observed_at_utc", ""),
        "meeting_date": meeting_date,
        "venue": venue,
        "race_no": race_no if race_no is not None else "",
        "pool_id": pool_id,
        "status": pool_pool.get("status") or odds_pool.get("status") or "",
        "sell_status": pool_pool.get("sellStatus") or odds_pool.get("sellStatus") or "",
        "leg_races": leg_races,
        "investment": investment,
        "odds_source_raw": odds_source_raw,
        "pool_source_raw": pool_source_raw,
        "odds_source": odds_source,
        "pool_source": pool_source,
        "odds_nodes": odds_pool.get("oddsNodes") or [],
        "initial_flags": flags,
        "error_message": " | ".join(error_parts),
    }


class MoneyFlowProcessor:
    def __init__(
        self,
        store: OutputStore,
        payout_rate: float = DEFAULT_PAYOUT_RATE,
        odds_divisor: float = 1.0,
        max_source_gap_seconds: float = 120.0,
        stale_after_seconds: float = 300.0,
        closure_tolerance: float = 0.20,
    ):
        if not 0 < payout_rate <= 1:
            raise ValueError("payout_rate must be in (0, 1]")
        if odds_divisor <= 0:
            raise ValueError("odds_divisor must be positive")
        self.store = store
        self.payout_rate = payout_rate
        self.odds_divisor = odds_divisor
        self.max_source_gap_seconds = max_source_gap_seconds
        self.stale_after_seconds = stale_after_seconds
        self.closure_tolerance = closure_tolerance
        self.seen_states = store.known_state_keys()
        self.previous_runners: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.last_runner_sets: dict[tuple[Any, ...], set[str]] = {}
        self._load_history()

    @staticmethod
    def race_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row.get("meeting_date", ""), row.get("venue", ""), str(row.get("race_no", "")))

    def _load_history(self) -> None:
        """Resume interval comparisons when a live collector is restarted."""
        if not self.store.runners_path.exists():
            return
        snapshot_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        with self.store.runners_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                stake = to_float(row.get("implied_stake"))
                observed = parse_timestamp(row.get("observed_at_utc"))
                if stake is None or observed is None:
                    continue
                key = self.race_key(row) + (row.get("runner_no", ""),)
                previous = self.previous_runners.get(key)
                if previous is None or observed >= previous["observed"]:
                    self.previous_runners[key] = {
                        "observed": observed,
                        "observed_at_utc": row.get("observed_at_utc", ""),
                        "snapshot_id": row.get("snapshot_id", ""),
                        "pool_id": row.get("pool_id", ""),
                        "investment": to_float(row.get("pool_investment")),
                        "decimal_odds": to_float(row.get("decimal_odds")),
                        "stake": stake,
                        "share": to_float(row.get("implied_stake_share")),
                    }
                race_key = self.race_key(row)
                group = snapshot_groups.setdefault(
                    race_key, {"observed": observed, "snapshot_id": "", "runners": set()}
                )
                if observed > group["observed"] or not group["snapshot_id"]:
                    group.update(
                        {
                            "observed": observed,
                            "snapshot_id": row.get("snapshot_id", ""),
                            "runners": set(),
                        }
                    )
                if group["snapshot_id"] == row.get("snapshot_id", ""):
                    group["runners"].add(row.get("runner_no", ""))
        for race_key, group in snapshot_groups.items():
            self.last_runner_sets[race_key] = group["runners"]

    def process(self, record: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        n = normalize_record(record)
        observed: datetime = n["observed"]
        flags = set(n["initial_flags"])
        if n["is_synthetic"]:
            flags.add("SYNTHETIC_DATA")
        elif not n["odds_convention_verified"]:
            flags.add("ODDS_CONVENTION_UNVERIFIED")
        if not n["odds_nodes"]:
            flags.add("EMPTY_ODDS_NODES")
        if n["investment"] is None:
            flags.add("MISSING_INVESTMENT")
        elif n["investment"] <= 0:
            flags.add("NONPOSITIVE_INVESTMENT")
        if str(n["sell_status"]).upper() == "STOP_SELL":
            flags.add("NOT_SELLING")

        odds_source: datetime | None = n["odds_source"]
        pool_source: datetime | None = n["pool_source"]
        if odds_source is None:
            flags.add("MISSING_ODDS_SOURCE_TIME")
        if pool_source is None:
            flags.add("MISSING_POOL_SOURCE_TIME")
        source_gap = None
        if odds_source is not None and pool_source is not None:
            source_gap = abs((odds_source - pool_source).total_seconds())
            if source_gap > self.max_source_gap_seconds:
                flags.add("SOURCE_TIMESTAMP_MISMATCH")
        for source, prefix in ((odds_source, "ODDS"), (pool_source, "POOL")):
            if source is None:
                continue
            age = (observed - source).total_seconds()
            if self.stale_after_seconds > 0 and age > self.stale_after_seconds:
                flags.add(f"STALE_{prefix}_SOURCE")
            if age < -60:
                flags.add(f"{prefix}_SOURCE_TIME_IN_FUTURE")

        state_material = {
            "meeting_date": n["meeting_date"],
            "venue": n["venue"],
            "race_no": n["race_no"],
            "pool_id": n["pool_id"],
            "investment": n["investment"],
            "odds": [
                {
                    "combString": node.get("combString"),
                    "oddsValue": node.get("oddsValue"),
                }
                for node in n["odds_nodes"]
                if isinstance(node, dict)
            ],
        }
        state_key = hashlib.sha256(json_text(state_material).encode("utf-8")).hexdigest()
        snapshot_id = hashlib.sha256(
            f"{state_key}|{n['record_id']}|{format_utc(observed)}".encode("utf-8")
        ).hexdigest()[:20]
        base = {
            "snapshot_id": snapshot_id,
            "record_id": n["record_id"],
            "state_key": state_key,
            "observed_at_utc": format_utc(observed),
            "odds_observed_at_utc": n["odds_observed_at_utc"],
            "pool_observed_at_utc": n["pool_observed_at_utc"],
            "source": n["source"],
            "is_synthetic": int(n["is_synthetic"]),
            "meeting_date": n["meeting_date"],
            "venue": n["venue"],
            "race_no": n["race_no"],
            "pool_id": n["pool_id"],
            "pool_status": n["status"],
            "sell_status": n["sell_status"],
            "leg_races": json_text(n["leg_races"]),
            "pool_investment": n["investment"] if n["investment"] is not None else "",
            "odds_source_time_raw": n["odds_source_raw"],
            "pool_source_time_raw": n["pool_source_raw"],
            "odds_source_time_utc": format_utc(odds_source),
            "pool_source_time_utc": format_utc(pool_source),
            "source_gap_seconds": source_gap if source_gap is not None else "",
            "runner_count": len(n["odds_nodes"]),
            "payout_rate": self.payout_rate,
            "odds_divisor": self.odds_divisor,
            "error_message": n["error_message"],
        }

        if state_key in self.seen_states:
            flags.add("UNCHANGED_STATE")
            base.update(
                {
                    "valid_odds_count": 0,
                    "calculation_status": "duplicate_unchanged_state",
                    "is_duplicate": 1,
                    "quality_flags": "|".join(sorted(flags)),
                }
            )
            return base, []
        self.seen_states.add(state_key)

        runner_rows: list[dict[str, Any]] = []
        local_flags: list[set[str]] = []
        valid_odds_count = 0
        calculated_count = 0
        investment_valid = n["investment"] is not None and n["investment"] > 0
        for node in n["odds_nodes"]:
            if not isinstance(node, dict):
                continue
            comb = node.get("combString")
            raw_odds = node.get("oddsValue")
            numeric_odds = to_float(raw_odds)
            decimal_odds = None
            row_flags: set[str] = set()
            if numeric_odds is None or numeric_odds <= 0:
                row_flags.add("INVALID_ODDS")
            else:
                decimal_odds = numeric_odds / self.odds_divisor
                valid_odds_count += 1
            stake = share = None
            if decimal_odds is not None and investment_valid:
                stake = self.payout_rate * n["investment"] / decimal_odds
                share = stake / n["investment"]
                calculated_count += 1
            runner_rows.append(
                {
                    **{key: base[key] for key in (
                        "snapshot_id", "record_id", "state_key", "observed_at_utc",
                        "source", "is_synthetic", "meeting_date", "venue", "race_no",
                        "pool_id", "pool_investment", "payout_rate",
                    )},
                    "runner_no": runner_number(comb),
                    "comb_string": "" if comb is None else str(comb),
                    "raw_odds_value": "" if raw_odds is None else str(raw_odds),
                    "decimal_odds": decimal_odds if decimal_odds is not None else "",
                    "hot_favourite": node.get("hotFavourite", ""),
                    "odds_drop_value": node.get("oddsDropValue", ""),
                    "implied_stake": stake if stake is not None else "",
                    "implied_stake_share": share if share is not None else "",
                    "flow_status": "pending",
                    "is_negative_flow": "",
                }
            )
            local_flags.append(row_flags)

        if valid_odds_count < len(runner_rows):
            flags.add("PARTIAL_ODDS_INVALID")
        sum_stake = sum(
            float(row["implied_stake"]) for row in runner_rows if row["implied_stake"] != ""
        ) if calculated_count else None
        closure = closure_relative = None
        if sum_stake is not None and investment_valid:
            closure = sum_stake - n["investment"]
            closure_relative = closure / n["investment"]
            if abs(closure_relative) > self.closure_tolerance:
                flags.add("POOL_CLOSURE_LARGE")

        race_key = self.race_key(n)
        current_runners = {row["runner_no"] for row in runner_rows}
        previous_runner_set = self.last_runner_sets.get(race_key, set())
        if previous_runner_set:
            if previous_runner_set - current_runners:
                flags.add("RUNNERS_MISSING")
            if current_runners - previous_runner_set:
                flags.add("RUNNERS_ADDED")

        for row, row_flags in zip(runner_rows, local_flags):
            key = race_key + (row["runner_no"],)
            previous = self.previous_runners.get(key)
            if row["implied_stake"] == "":
                row["flow_status"] = "not_calculated"
            elif previous is None:
                row["flow_status"] = "first_valid_snapshot"
            else:
                row.update(
                    {
                        "previous_snapshot_id": previous["snapshot_id"],
                        "previous_observed_at_utc": previous["observed_at_utc"],
                        "previous_pool_investment": previous["investment"] if previous["investment"] is not None else "",
                        "previous_decimal_odds": previous["decimal_odds"] if previous["decimal_odds"] is not None else "",
                        "previous_implied_stake": previous["stake"],
                        "previous_implied_stake_share": previous["share"] if previous["share"] is not None else "",
                    }
                )
                interval = (observed - previous["observed"]).total_seconds()
                row["interval_seconds"] = interval
                if previous["pool_id"] and n["pool_id"] and previous["pool_id"] != n["pool_id"]:
                    row["flow_status"] = "not_comparable_pool_changed"
                    row_flags.add("POOL_ID_CHANGED")
                elif interval <= 0:
                    row["flow_status"] = "not_calculated_non_chronological"
                    row_flags.add("NON_CHRONOLOGICAL_SNAPSHOT")
                else:
                    stake = float(row["implied_stake"])
                    share = float(row["implied_stake_share"])
                    flow = stake - previous["stake"]
                    share_change = share - previous["share"] if previous["share"] is not None else None
                    row["pool_investment_change"] = (
                        n["investment"] - previous["investment"]
                        if previous["investment"] is not None
                        else ""
                    )
                    row["implied_stake_flow"] = flow
                    row["implied_stake_share_change"] = (
                        share_change if share_change is not None else ""
                    )
                    row["implied_flow_per_minute"] = flow / (interval / 60.0)
                    row["odds_changed"] = int(
                        previous["decimal_odds"] is None
                        or not math.isclose(
                            float(row["decimal_odds"]), previous["decimal_odds"], abs_tol=1e-12
                        )
                    )
                    row["is_negative_flow"] = int(flow < 0)
                    row["flow_status"] = "estimated_negative_flow" if flow < 0 else "estimated"
                    if flow < 0:
                        row_flags.add("NEGATIVE_IMPLIED_FLOW")
            if row["implied_stake"] != "" and (
                previous is None or observed > previous["observed"]
            ):
                self.previous_runners[key] = {
                    "observed": observed,
                    "observed_at_utc": row["observed_at_utc"],
                    "snapshot_id": row["snapshot_id"],
                    "pool_id": row["pool_id"],
                    "investment": n["investment"],
                    "decimal_odds": decimal_odds_from_row(row),
                    "stake": float(row["implied_stake"]),
                    "share": to_float(row["implied_stake_share"]),
                }
            row["quality_flags"] = "|".join(sorted(flags | row_flags))

        self.last_runner_sets[race_key] = current_runners
        if any("NEGATIVE_IMPLIED_FLOW" in row["quality_flags"] for row in runner_rows):
            flags.add("NEGATIVE_IMPLIED_FLOW")
        if calculated_count == 0:
            status = "no_market_data" if not runner_rows else "not_calculated"
        else:
            benign = {"SYNTHETIC_DATA", "ODDS_CONVENTION_UNVERIFIED"}
            status = "estimated_with_warnings" if flags - benign else "estimated"
        base.update(
            {
                "valid_odds_count": valid_odds_count,
                "sum_implied_stake": sum_stake if sum_stake is not None else "",
                "sum_implied_stake_share": (
                    sum_stake / n["investment"] if sum_stake is not None and investment_valid else ""
                ),
                "pool_closure_error": closure if closure is not None else "",
                "pool_closure_relative": closure_relative if closure_relative is not None else "",
                "calculation_status": status,
                "is_duplicate": 0,
                "quality_flags": "|".join(sorted(flags)),
            }
        )
        return base, runner_rows


def decimal_odds_from_row(row: dict[str, Any]) -> float | None:
    return to_float(row.get("decimal_odds"))


class OutputWriter:
    def __init__(self, store: OutputStore, processor: MoneyFlowProcessor, quiet: bool = False):
        self.store = store
        self.processor = processor
        self.quiet = quiet
        self.counts = {"records": 0, "snapshots": 0, "runners": 0, "duplicates": 0}

    def handle(self, record: dict[str, Any]) -> dict[str, Any]:
        record.setdefault("record_id", f"record-{uuid.uuid4()}")
        self.store.append_raw(record)
        snapshot, runners = self.processor.process(record)
        self.store.append_snapshot(snapshot)
        for row in runners:
            self.store.append_runner(row)
        event = {
            "event_at_utc": utc_now(),
            "record_id": snapshot["record_id"],
            "snapshot_id": snapshot["snapshot_id"],
            "calculation_status": snapshot["calculation_status"],
            "runner_rows": len(runners),
            "quality_flags": snapshot["quality_flags"],
        }
        self.store.append_event(event)
        self.counts["records"] += 1
        self.counts["snapshots"] += 1
        self.counts["runners"] += len(runners)
        self.counts["duplicates"] += int(snapshot["is_duplicate"])
        if not self.quiet:
            label = (
                f"{snapshot['meeting_date'] or 'unknown-date'} "
                f"{snapshot['venue']} R{snapshot['race_no']}"
            )
            print(
                f"[{snapshot['observed_at_utc']}] {label}: "
                f"{snapshot['calculation_status']}; runners={len(runners)}; "
                f"flags={snapshot['quality_flags'] or 'none'}"
            )
        return snapshot


def load_offline_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8-sig")
    if input_path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        document = json.loads(text)
        if isinstance(document, list):
            records = document
        elif isinstance(document, dict) and "snapshots" in document:
            records = document["snapshots"]
            for record in records:
                if "is_synthetic" not in record and "is_synthetic" in document:
                    record["is_synthetic"] = document["is_synthetic"]
        elif isinstance(document, dict):
            records = [document]
        else:
            raise ValueError("Offline input must be a JSON object, array, or JSONL records")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every offline snapshot must be a JSON object")
    for index, record in enumerate(records, start=1):
        record.setdefault("record_id", f"offline-{index:06d}")
        record.setdefault("source", "OFFLINE")
    return records


def add_quality_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--payout-rate", type=float, default=DEFAULT_PAYOUT_RATE,
                        help="Dividend payout rate used by the approximation (default: 0.825)")
    parser.add_argument("--odds-divisor", type=float, default=1.0,
                        help="Divide raw oddsValue by this number before calculation (default: 1)")
    parser.add_argument("--max-source-gap-seconds", type=float, default=120.0,
                        help="Flag odds/pool source timestamps farther apart than this")
    parser.add_argument("--stale-after-seconds", type=float, default=300.0,
                        help="Flag source timestamps older than this; 0 disables the age check")
    parser.add_argument("--closure-tolerance", type=float, default=0.20,
                        help="Flag absolute pool-closure relative error above this (default: 0.20)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-record console output")


def build_writer(args: argparse.Namespace) -> OutputWriter:
    store = OutputStore(args.output_dir)
    processor = MoneyFlowProcessor(
        store,
        payout_rate=args.payout_rate,
        odds_divisor=args.odds_divisor,
        max_source_gap_seconds=args.max_source_gap_seconds,
        stale_after_seconds=args.stale_after_seconds,
        closure_tolerance=args.closure_tolerance,
    )
    return OutputWriter(store, processor, quiet=args.quiet)


def run_offline(args: argparse.Namespace) -> int:
    writer = build_writer(args)
    records = load_offline_records(args.input)
    for record in records:
        writer.handle(record)
    print(f"Finished: {json_text(writer.counts)}; output={Path(args.output_dir).resolve()}")
    return 0


def run_live(args: argparse.Namespace) -> int:
    writer = build_writer(args)
    client = HKJCClient(timeout=args.timeout, retries=args.retries)
    poll = 0
    try:
        while args.max_polls == 0 or poll < args.max_polls:
            poll += 1
            for race_no in args.races:
                record = client.fetch_race(race_no, args.venue.upper(), args.date)
                writer.handle(record)
            if args.max_polls != 0 and poll >= args.max_polls:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Collection stopped by user; existing append-only files were retained.")
    print(f"Finished: {json_text(writer.counts)}; output={Path(args.output_dir).resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Timestamped HKJC WIN odds and approximate runner-level money-flow reconstruction"
    )
    parser.add_argument("--version", action="version", version=VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    live = subparsers.add_parser("live", help="Collect read-only public GraphQL snapshots")
    live.add_argument("--venue", default="ST", choices=["ST", "HV"], help="Request venue label")
    live.add_argument("--races", type=int, nargs="+", default=[1], help="Race number(s)")
    live.add_argument("--date", default=None, help="Optional requested meeting date, YYYY-MM-DD")
    live.add_argument("--interval", type=float, default=60.0,
                      help="Seconds between polling rounds (default: 60)")
    live.add_argument("--max-polls", type=int, default=1,
                      help="Polling rounds; 0 means run until Ctrl+C (default: 1)")
    live.add_argument("--timeout", type=float, default=15.0)
    live.add_argument("--retries", type=int, default=3)
    live.add_argument("--output-dir", default="live_output")
    add_quality_arguments(live)
    live.set_defaults(function=run_live)

    offline = subparsers.add_parser("offline", help="Process a JSON or JSONL snapshot file")
    offline.add_argument("--input", required=True, help="Offline JSON/JSONL input")
    offline.add_argument("--output-dir", required=True)
    add_quality_arguments(offline)
    offline.set_defaults(function=run_offline)

    demo = subparsers.add_parser("demo", help="Run the included explicitly synthetic example")
    demo.add_argument("--output-dir", default="demo_output")
    add_quality_arguments(demo)
    demo.set_defaults(function=run_offline, input=str(Path(__file__).parent / "examples" / "synthetic_snapshots.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "interval", 60) < 0 or getattr(args, "max_polls", 1) < 0:
        parser.error("interval and max-polls must be nonnegative")
    if any(race < 1 for race in getattr(args, "races", [1])):
        parser.error("race numbers must be positive")
    try:
        return int(args.function(args))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
