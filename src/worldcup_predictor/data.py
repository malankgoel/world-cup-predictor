from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)
CUP_URL = (
    "https://raw.githubusercontent.com/openfootball/"
    "worldcup/master/2026--usa/cup.txt"
)
FINALS_URL = (
    "https://raw.githubusercontent.com/openfootball/"
    "worldcup/master/2026--usa/cup_finals.txt"
)

TEAM_ALIASES = {
    "USA": "United States",
    "United States of America": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
}

HOST_COUNTRY = {
    "Mexico": "Mexico",
    "Canada": "Canada",
    "United States": "United States",
}

CITY_COUNTRY = {
    "Mexico City": "Mexico",
    "Guadalajara (Zapopan)": "Mexico",
    "Monterrey (Guadalupe)": "Mexico",
    "Toronto": "Canada",
    "Vancouver": "Canada",
}

RESULT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]


def normalize_team(value: str) -> str:
    name = str(value).strip()
    return TEAM_ALIASES.get(name, name)


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def download_results(path: str | Path) -> Path:
    response = requests.get(RESULTS_URL, timeout=30)
    response.raise_for_status()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def load_results(path: str | Path, start_date: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(RESULT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Results file is missing columns: {sorted(missing)}")
    if "winner" not in frame:
        frame["winner"] = ""
    frame = frame[RESULT_COLUMNS + ["winner"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["home_team"] = frame["home_team"].map(normalize_team)
    frame["away_team"] = frame["away_team"].map(normalize_team)
    frame["winner"] = frame["winner"].fillna("").map(
        lambda value: normalize_team(value) if value else ""
    )
    frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
    frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
    frame = frame.dropna(subset=["home_score", "away_score"])
    frame["home_score"] = frame["home_score"].astype(int)
    frame["away_score"] = frame["away_score"].astype(int)
    frame["neutral"] = frame["neutral"].map(_bool)
    if start_date:
        frame = frame[frame["date"] >= pd.Timestamp(start_date)]
    return frame.sort_values("date").reset_index(drop=True)


def load_optional(path: str | Path, columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def _venue_country(venue: str) -> str:
    return CITY_COUNTRY.get(venue, "United States")


def _parse_schedule(text: str, knockout: bool) -> list[dict]:
    rows = []
    current_group = ""
    stage = "group"
    current_date = None
    next_group_id = 1

    date_re = re.compile(
        r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) "
        r"(?:June|Jun|July|Jul) \d{1,2}$"
    )
    match_re = re.compile(
        r"^\s*(?:\((\d+)\)\s*)?"
        r"(\d{1,2}:\d{2})\s+UTC([+-]\d+)\s+(.+?)\s+v\s+(.+?)\s+@\s+(.+?)\s*$"
    )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("▪ Group "):
            current_group = line.removeprefix("▪ Group ").strip()
            stage = "group"
        elif line.startswith("▪ Round of 32"):
            stage = "round_of_32"
        elif line.startswith("▪ Round of 16"):
            stage = "round_of_16"
        elif line.startswith("▪ Quarter-final"):
            stage = "quarter_final"
        elif line.startswith("▪ Semi-final"):
            stage = "semi_final"
        elif line.startswith("▪ Match for third place"):
            stage = "third_place"
        elif line.startswith("▪ Final"):
            stage = "final"
        elif date_re.match(line):
            current_date = pd.to_datetime(f"{line} 2026").date().isoformat()
        else:
            match = match_re.match(line)
            if not match or not current_date:
                continue
            match_id, time, utc_offset, home, away, venue = match.groups()
            if knockout:
                home_source, away_source = home.strip(), away.strip()
                home_team = away_team = ""
            else:
                match_id = str(next_group_id)
                next_group_id += 1
                home_team = normalize_team(home)
                away_team = normalize_team(away)
                home_source, away_source = home_team, away_team
            venue_country = _venue_country(venue)
            host_in_match = (
                HOST_COUNTRY.get(home_team) == venue_country
                or HOST_COUNTRY.get(away_team) == venue_country
            )
            rows.append(
                {
                    "match_id": int(match_id),
                    "date": current_date,
                    "time": time,
                    "utc_offset": int(utc_offset),
                    "stage": stage,
                    "group": current_group if stage == "group" else "",
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_source": home_source,
                    "away_source": away_source,
                    "venue": venue,
                    "venue_country": venue_country,
                    "neutral": not host_in_match,
                }
            )
    return rows


def download_schedule(path: str | Path) -> Path:
    cup = requests.get(CUP_URL, timeout=30)
    finals = requests.get(FINALS_URL, timeout=30)
    cup.raise_for_status()
    finals.raise_for_status()
    group_rows = pd.DataFrame(_parse_schedule(cup.text, knockout=False))
    kickoff_utc = (
        pd.to_datetime(group_rows["date"] + " " + group_rows["time"])
        - pd.to_timedelta(group_rows["utc_offset"], unit="h")
    )
    group_rows = group_rows.assign(_kickoff=kickoff_utc).sort_values("_kickoff")
    group_rows["match_id"] = range(1, len(group_rows) + 1)
    group_rows = group_rows.drop(columns="_kickoff")
    knockout_rows = pd.DataFrame(_parse_schedule(finals.text, knockout=True))
    frame = pd.concat([group_rows, knockout_rows], ignore_index=True).sort_values(
        "match_id"
    )
    if len(frame) != 104:
        raise ValueError(f"Expected 104 World Cup matches, parsed {len(frame)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_schedule(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "match_id",
        "date",
        "stage",
        "group",
        "home_team",
        "away_team",
        "home_source",
        "away_source",
        "venue_country",
        "neutral",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Schedule is missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["home_team"] = frame["home_team"].map(normalize_team)
    frame["away_team"] = frame["away_team"].map(normalize_team)
    frame["neutral"] = frame["neutral"].map(_bool)
    return frame.sort_values("match_id").reset_index(drop=True)
