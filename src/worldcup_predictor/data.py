from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/shootouts.csv"
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

# Approximate venue attributes used for 2026 schedule features. Coordinates and
# altitude are intentionally coarse: the model needs the material differences
# (Mexico City altitude and cross-continent travel), not stadium-survey precision.
VENUE_ATTRIBUTES = {
    "Atlanta": (33.76, -84.40, 320.0, 29.0, 68.0),
    "Boston": (42.09, -71.26, 50.0, 25.0, 66.0),
    "Boston (Foxborough)": (42.09, -71.26, 50.0, 25.0, 66.0),
    "Dallas": (32.75, -97.09, 190.0, 32.0, 61.0),
    "Dallas (Arlington)": (32.75, -97.09, 190.0, 32.0, 61.0),
    "East Rutherford": (40.81, -74.07, 2.0, 27.0, 65.0),
    "Guadalajara": (20.66, -103.35, 1560.0, 29.0, 55.0),
    "Guadalajara (Zapopan)": (20.68, -103.46, 1570.0, 29.0, 55.0),
    "Houston": (29.68, -95.41, 15.0, 32.0, 75.0),
    "Kansas City": (39.05, -94.48, 270.0, 29.0, 67.0),
    "Los Angeles": (33.95, -118.34, 40.0, 25.0, 62.0),
    "Los Angeles (Inglewood)": (33.95, -118.34, 40.0, 25.0, 62.0),
    "Mexico City": (19.30, -99.15, 2240.0, 24.0, 55.0),
    "Miami": (25.96, -80.24, 2.0, 31.0, 75.0),
    "Miami (Miami Gardens)": (25.96, -80.24, 2.0, 31.0, 75.0),
    "Monterrey": (25.67, -100.24, 540.0, 34.0, 58.0),
    "Monterrey (Guadalupe)": (25.67, -100.24, 540.0, 34.0, 58.0),
    "Philadelphia": (39.90, -75.17, 12.0, 28.0, 67.0),
    "San Francisco": (37.40, -121.97, 12.0, 23.0, 62.0),
    "San Francisco Bay Area (Santa Clara)": (
        37.40,
        -121.97,
        12.0,
        23.0,
        62.0,
    ),
    "Seattle": (47.60, -122.33, 20.0, 22.0, 65.0),
    "Toronto": (43.63, -79.42, 75.0, 25.0, 65.0),
    "Vancouver": (49.28, -123.11, 5.0, 21.0, 68.0),
    "New York/New Jersey (East Rutherford)": (
        40.81,
        -74.07,
        2.0,
        27.0,
        65.0,
    ),
}

# Current national-team confederations. Historical aliases are normalized
# before lookup. Unknown teams remain missing rather than being guessed.
CONFEDERATION = {
    # AFC
    **{
        team: "AFC"
        for team in (
            "Australia",
            "China PR",
            "Iran",
            "Iraq",
            "Japan",
            "Jordan",
            "North Korea",
            "Qatar",
            "Saudi Arabia",
            "South Korea",
            "United Arab Emirates",
            "Uzbekistan",
        )
    },
    # CAF
    **{
        team: "CAF"
        for team in (
            "Algeria",
            "Cameroon",
            "Cape Verde",
            "DR Congo",
            "Egypt",
            "Ghana",
            "Ivory Coast",
            "Mali",
            "Morocco",
            "Nigeria",
            "Senegal",
            "South Africa",
            "Tunisia",
        )
    },
    # Concacaf
    **{
        team: "CONCACAF"
        for team in (
            "Canada",
            "Costa Rica",
            "Curaçao",
            "Haiti",
            "Honduras",
            "Jamaica",
            "Mexico",
            "Panama",
            "Trinidad and Tobago",
            "United States",
        )
    },
    # CONMEBOL
    **{
        team: "CONMEBOL"
        for team in (
            "Argentina",
            "Bolivia",
            "Brazil",
            "Chile",
            "Colombia",
            "Ecuador",
            "Paraguay",
            "Peru",
            "Uruguay",
            "Venezuela",
        )
    },
    # OFC
    **{team: "OFC" for team in ("New Caledonia", "New Zealand", "Tahiti")},
    # UEFA
    **{
        team: "UEFA"
        for team in (
            "Austria",
            "Belgium",
            "Bosnia and Herzegovina",
            "Croatia",
            "Czech Republic",
            "Denmark",
            "England",
            "France",
            "Germany",
            "Italy",
            "Netherlands",
            "Norway",
            "Poland",
            "Portugal",
            "Scotland",
            "Serbia",
            "Spain",
            "Sweden",
            "Switzerland",
            "Turkey",
            "Ukraine",
            "Wales",
        )
    },
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

OPTIONAL_RESULT_COLUMNS = [
    "winner",
    "stage",
    "home_xg",
    "away_xg",
    "home_odds",
    "draw_odds",
    "away_odds",
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


def download_shootouts(path: str | Path) -> Path:
    response = requests.get(SHOOTOUTS_URL, timeout=30)
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
    keep = RESULT_COLUMNS + [
        column for column in OPTIONAL_RESULT_COLUMNS if column in frame.columns
    ]
    frame = frame[keep].copy()
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
    for column in ("home_xg", "away_xg", "home_odds", "draw_odds", "away_odds"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
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
