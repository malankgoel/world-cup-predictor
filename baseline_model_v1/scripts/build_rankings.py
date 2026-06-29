"""Build dated FIFA ranking snapshots from public historical archives.

The long archive supplies official-release snapshots through 2018. A current
international-match feature archive supplies the pre-match FIFA points carried
by each team from 2019 onward, extending coverage through March 2026.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldcup_predictor.data import normalize_team  # noqa: E402

HISTORICAL_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now"
)
CURRENT_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "arrnalireza/international-football-dataset-from-2004-to-2026"
)
OUTPUT = ROOT / "data" / "input" / "rankings.csv"
SOURCES = ROOT / "data" / "input" / "rankings_sources.txt"


def _archive(url: str, local_path: str | None) -> zipfile.ZipFile:
    if local_path:
        return zipfile.ZipFile(local_path)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(response.content))


def historical(path: str | None) -> pd.DataFrame:
    with _archive(HISTORICAL_URL, path) as archive:
        frame = pd.read_csv(archive.open("fifa_ranking.csv"))
    frame = frame.rename(
        columns={
            "rank_date": "date",
            "country_full": "team",
            "total_points": "points",
        }
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.loc[
        frame["date"] < pd.Timestamp("2019-01-01"),
        ["date", "team", "rank", "points"],
    ]


def current(path: str | None) -> pd.DataFrame:
    filename = "national_matches_(2004-2026)_v1.csv"
    with _archive(CURRENT_URL, path) as archive:
        frame = pd.read_csv(archive.open(filename))
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["date"] >= pd.Timestamp("2019-01-01")]
    home = frame[
        ["date", "home_team", "home_rank", "home_fifa_points"]
    ].rename(
        columns={
            "home_team": "team",
            "home_rank": "rank",
            "home_fifa_points": "points",
        }
    )
    away = frame[
        ["date", "away_team", "away_rank", "away_fifa_points"]
    ].rename(
        columns={
            "away_team": "team",
            "away_rank": "rank",
            "away_fifa_points": "points",
        }
    )
    return pd.concat([home, away], ignore_index=True)


def main(
    historical_zip: str | None = None,
    current_zip: str | None = None,
) -> None:
    frame = pd.concat(
        [historical(historical_zip), current(current_zip)],
        ignore_index=True,
    )
    frame["team"] = frame["team"].map(normalize_team)
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["points"] = pd.to_numeric(frame["points"], errors="coerce")
    frame = (
        frame.dropna(subset=["date", "team", "rank", "points"])
        .sort_values(["team", "date"])
        .drop_duplicates(["date", "team"], keep="last")
        .sort_values(["date", "rank", "team"])
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False, date_format="%Y-%m-%d")
    SOURCES.write_text(
        "Historical monthly releases through 2018:\n"
        f"{HISTORICAL_URL}\n\n"
        "Pre-match ranking points from 2019 through March 2026:\n"
        f"{CURRENT_URL}\n"
    )
    print(
        f"wrote {len(frame):,} snapshots for {frame['team'].nunique()} teams "
        f"from {frame['date'].min().date()} through {frame['date'].max().date()}"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
