#!/usr/bin/env python3
"""Collect Hong Kong race results from official HKJC pages for academic analysis."""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
DATA = ROOT / "data"
BASE = "https://racing.hkjc.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AcademicCapstoneResearch/1.0)"}


def fetch(url: str, cache_key: str | None = None, retries: int = 4) -> str:
    path = CACHE / f"{cache_key}.html.gz" if cache_key else None
    if path and path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=40)
            response.raise_for_status()
            text = response.text
            if len(text) < 10000:
                raise ValueError(f"Unexpectedly short page ({len(text)} chars)")
            if path:
                tmp = path.with_suffix(".tmp")
                with gzip.open(tmp, "wt", encoding="utf-8") as f:
                    f.write(text)
                tmp.replace(path)
            return text
        except Exception as exc:
            error = exc
            time.sleep((2 ** attempt) + random.random())
    raise RuntimeError(f"Failed {url}: {error}")


def season_months(first_year: int):
    for year, month in [(first_year, m) for m in range(9, 13)] + [(first_year + 1, m) for m in range(1, 8)]:
        yield year, month


def get_fixtures(seasons: list[int]) -> pd.DataFrame:
    records = []
    for season in seasons:
        for year, month in season_months(season):
            url = f"{BASE}/racing/information/English/Racing/Fixture.aspx?CalYear={year}&CalMonth={month:02d}"
            html = fetch(url, f"fixture_{year}_{month:02d}")
            soup = BeautifulSoup(html, "html.parser")
            for cell in soup.select("td.calendar"):
                day_el = cell.select_one("span.f_fl.f_fs14")
                alts = [img.get("alt", "") for img in cell.find_all("img")]
                course = next((x for x in alts if x in ("ST", "HV")), None)
                if not day_el or not course:
                    continue
                day = int(day_el.get_text(strip=True))
                date = f"{year}-{month:02d}-{day:02d}"
                race_count = sum(int(n) for _, n in re.findall(r"(\d+)\((\d+)\)", cell.get_text(" ", strip=True)))
                records.append({"date": date, "course": course, "season": f"{season}/{str(season + 1)[-2:]}",
                                "fixture_races": race_count, "fixture_url": url})
    frame = pd.DataFrame(records).drop_duplicates(["date", "course"], keep="last")
    return frame.sort_values(["date", "course"]).reset_index(drop=True)


def cells(row):
    return [" ".join(c.stripped_strings) for c in row.find_all(["td", "th"], recursive=False)]


def parse_result(html: str, date: str, course: str, season: str, race_no: int, url: str):
    soup = BeautifulSoup(html, "html.parser")
    runner_table = None
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if rows and "Win Odds" in cells(rows[0]) and "Horse No." in cells(rows[0]) and "Finish Time" in cells(rows[0]):
            runner_table = table
            break
    if runner_table is None:
        return None

    tables = soup.find_all("table")
    runner_index = tables.index(runner_table)
    meta_table = tables[runner_index - 1]
    meta_rows = [cells(r) for r in meta_table.find_all("tr")]
    meta_text = [" | ".join(r) for r in meta_rows]
    class_text = next((r[0] for r in meta_rows if r and re.search(r"\d+M", r[0])), "")
    race_name = ""
    going = ""
    course_config = ""
    purse = ""
    for row in meta_rows:
        if not row:
            continue
        if "Going :" in row:
            going = row[-1]
        elif "Course :" in row:
            course_config = row[-1]
        elif row[0].startswith("HK$"):
            purse = row[0]
        elif row[0] and row[0] != class_text and not race_name and not row[0].startswith("RACE"):
            race_name = row[0]
    distance_match = re.search(r"(\d+)M", class_text, re.I)
    distance = int(distance_match.group(1)) if distance_match else None
    class_name = class_text.split("-")[0].strip() if class_text else ""
    rating_band = ""
    band_match = re.search(r"\((\d+-\d+)\)", class_text)
    if band_match:
        rating_band = band_match.group(1)

    race_id = f"{date.replace('-', '')}_{course}_{race_no:02d}"
    race = {
        "race_id": race_id, "date": date, "course": course, "season": season, "race_no": race_no,
        "class_text": class_text, "class_name": class_name, "distance_m": distance,
        "rating_band": rating_band, "race_name": race_name, "going": going,
        "course_config": course_config, "purse": purse, "source_url": url,
    }

    runners = []
    headers = cells(runner_table.find_all("tr")[0])
    for row in runner_table.find_all("tr")[1:]:
        values = cells(row)
        if len(values) < len(headers) or not values[1].isdigit():
            continue
        item = dict(zip(headers, values))
        horse_text = item.get("Horse", "")
        id_match = re.search(r"\(([A-Z]\d{3})\)", horse_text)
        horse_id = id_match.group(1) if id_match else re.sub(r"\W+", "", horse_text)
        horse_name = re.sub(r"\s*\([A-Z]\d{3}\)\s*$", "", horse_text).strip()
        finish_time = item.get("Finish Time", "")
        seconds = None
        time_match = re.match(r"(?:(\d+):)?(\d+)\.(\d{2})$", finish_time)
        if time_match:
            seconds = int(time_match.group(1) or 0) * 60 + int(time_match.group(2)) + int(time_match.group(3)) / 100
        runners.append({
            "race_id": race_id, "date": date, "course": course, "season": season, "race_no": race_no,
            "placing_raw": item.get("Pla.", ""), "horse_no": int(item.get("Horse No.")),
            "horse_id": horse_id, "horse_name": horse_name, "jockey": item.get("Jockey", ""),
            "trainer": item.get("Trainer", ""), "actual_weight_lbs": pd.to_numeric(item.get("Act. Wt."), errors="coerce"),
            "horse_weight_lbs": pd.to_numeric(item.get("Declar. Horse Wt."), errors="coerce"),
            "draw": pd.to_numeric(item.get("Dr."), errors="coerce"), "lbw": item.get("LBW", ""),
            "running_position": item.get("Running Position", ""), "finish_time": finish_time,
            "finish_seconds": seconds, "win_odds": pd.to_numeric(item.get("Win Odds"), errors="coerce"),
            "source_url": url,
        })

    dividends = {}
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) > 1 and "Dividend (HK$)" in cells(rows[1]):
            for row in rows[2:]:
                values = cells(row)
                if len(values) >= 3 and values[0].upper() == "WIN":
                    combo = values[1].replace(" ", "")
                    try:
                        dividends[combo] = float(values[2].replace(",", ""))
                    except ValueError:
                        pass
            break
    for runner in runners:
        runner["win_dividend_hkd10"] = dividends.get(str(runner["horse_no"]))
        runner["won"] = int(runner["placing_raw"] == "1")
    return race, runners


def race_jobs(fixtures: pd.DataFrame):
    for row in fixtures.itertuples(index=False):
        maximum = 12 if row.fixture_races == 0 else min(row.fixture_races + 1, 14)
        for race_no in range(1, maximum + 1):
            yield row, race_no


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    fixtures = get_fixtures(args.seasons)
    fixtures.to_csv(DATA / "fixtures.csv", index=False)
    jobs = list(race_jobs(fixtures))
    print(f"Meetings: {len(fixtures)}; candidate race pages: {len(jobs)}", flush=True)

    races, runners, errors = [], [], []
    found_by_meeting = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for row, race_no in jobs:
            date_url = row.date.replace("-", "/")
            url = (f"{BASE}/en-us/local/information/archive/localresults?racedate={date_url}"
                   f"&racecourse={row.course}&RaceNo={race_no}")
            race_id = f"{row.date.replace('-', '')}_{row.course}_{race_no:02d}"
            futures[pool.submit(fetch, url, race_id)] = (row, race_no, url)
        for count, future in enumerate(as_completed(futures), 1):
            row, race_no, url = futures[future]
            key = (row.date, row.course)
            try:
                html = future.result()
                parsed = parse_result(html, row.date, row.course, row.season, race_no, url)
                if parsed:
                    race, entries = parsed
                    races.append(race)
                    runners.extend(entries)
                    found_by_meeting[key] = found_by_meeting.get(key, 0) + 1
            except Exception as exc:
                errors.append({"date": row.date, "course": row.course, "race_no": race_no, "error": str(exc)})
            if count % 100 == 0 or count == len(futures):
                print(f"Processed {count}/{len(futures)}; races={len(races)}; runners={len(runners)}; errors={len(errors)}", flush=True)

    race_frame = pd.DataFrame(races).sort_values(["date", "course", "race_no"])
    runner_frame = pd.DataFrame(runners).sort_values(["date", "course", "race_no", "horse_no"])
    race_frame["field_size"] = race_frame["race_id"].map(runner_frame.groupby("race_id").size())
    race_frame["winners"] = race_frame["race_id"].map(runner_frame.groupby("race_id")["won"].sum())
    race_frame["valid_odds_runners"] = race_frame["race_id"].map(runner_frame.dropna(subset=["win_odds"]).groupby("race_id").size()).fillna(0).astype(int)
    race_frame["win_dividend"] = race_frame["race_id"].map(runner_frame.dropna(subset=["win_dividend_hkd10"]).groupby("race_id")["win_dividend_hkd10"].first())
    race_frame.to_csv(DATA / "races_raw.csv", index=False)
    runner_frame.to_csv(DATA / "runners_raw.csv", index=False)
    pd.DataFrame(errors).to_csv(DATA / "collection_errors.csv", index=False)

    coverage = fixtures.copy()
    coverage["collected_races"] = coverage.apply(lambda r: found_by_meeting.get((r["date"], r["course"]), 0), axis=1)
    coverage.to_csv(DATA / "coverage.csv", index=False)
    manifest = {
        "retrieved_utc": datetime.utcnow().isoformat() + "Z",
        "source": "The Hong Kong Jockey Club official local results and fixture pages",
        "seasons": [f"{s}/{str(s + 1)[-2:]}" for s in args.seasons],
        "meetings": len(fixtures), "races": len(race_frame), "runners": len(runner_frame),
        "candidate_pages": len(jobs), "errors": len(errors),
    }
    (DATA / "collection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
