"""Build annual national-team talent proxies from FIFA/EA player databases.

For each edition, the highest-rated 23 players by nationality form a dated
proxy squad. The existing June 2026 actual squad snapshot is retained and
supersedes these annual proxies for current predictions.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldcup_predictor.data import normalize_team  # noqa: E402

FIFA_15_22_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "stefanoleone992/fifa-22-complete-player-dataset"
)
FC24_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "stefanoleone992/ea-sports-fc-24-complete-player-dataset"
)
FC25_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "aniss7/fifa-player-data-from-sofifa-2025-06-03"
)
OUTPUT = ROOT / "data" / "input" / "squads.csv"
SOURCES = ROOT / "data" / "input" / "squads_sources.txt"


def _archive(url: str, local_path: str | None) -> zipfile.ZipFile:
    if local_path:
        return zipfile.ZipFile(local_path)
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(response.content))


def _position(value: str) -> str:
    first = str(value).split(",")[0].strip().upper()
    if first in {"GK", "G"}:
        return "GK"
    if first in {"CB", "LB", "RB", "LWB", "RWB", "SW", "DEF", "DF"}:
        return "DEF"
    if first in {
        "CDM",
        "CM",
        "CAM",
        "LM",
        "RM",
        "MID",
        "MF",
    }:
        return "MID"
    return "FWD"


def _select_squad(frame: pd.DataFrame, as_of: str) -> pd.DataFrame:
    frame = frame.dropna(subset=["team", "player", "talent"]).copy()
    frame["team"] = frame["team"].map(normalize_team)
    frame["position"] = frame["position"].map(_position)
    rows = []
    for _, candidates in frame.groupby("team"):
        candidates = candidates.sort_values("talent", ascending=False).drop_duplicates(
            "player"
        )
        selected = candidates.head(23).copy()
        if len(selected) < 11:
            continue
        starters = []
        for position, count in (("GK", 1), ("DEF", 4), ("MID", 4), ("FWD", 2)):
            starters.extend(
                selected[selected["position"] == position].head(count).index
            )
        if len(starters) < 11:
            starters.extend(
                selected[~selected.index.isin(starters)]
                .head(11 - len(starters))
                .index
            )
        selected["is_starter"] = selected.index.isin(starters)
        selected["exp_minutes"] = np.where(selected["is_starter"], 0.90, 0.25)
        selected["available"] = True
        selected["talent_se"] = 3.0
        selected["as_of"] = as_of
        rows.append(selected)
    columns = [
        "as_of",
        "team",
        "player",
        "club",
        "position",
        "talent",
        "talent_se",
        "exp_minutes",
        "available",
        "age",
        "is_starter",
    ]
    return pd.concat(rows, ignore_index=True)[columns]


def fifa_15_22(path: str | None) -> list[pd.DataFrame]:
    outputs = []
    with _archive(FIFA_15_22_URL, path) as archive:
        for edition in range(15, 23):
            source = pd.read_csv(
                archive.open(f"players_{edition}.csv"),
                usecols=[
                    "long_name",
                    "player_positions",
                    "overall",
                    "age",
                    "club_name",
                    "nationality_name",
                ],
            )
            source = source.rename(
                columns={
                    "long_name": "player",
                    "player_positions": "position",
                    "overall": "talent",
                    "club_name": "club",
                    "nationality_name": "team",
                }
            )
            outputs.append(
                _select_squad(source, f"20{edition - 1:02d}-09-01")
            )
    return outputs


def fc24(path: str | None) -> pd.DataFrame:
    with _archive(FC24_URL, path) as archive:
        source = pd.read_csv(
            archive.open("male_players.csv"),
            usecols=[
                "update_as_of",
                "long_name",
                "player_positions",
                "overall",
                "age",
                "club_name",
                "nationality_name",
            ],
        )
    source["update_as_of"] = pd.to_datetime(source["update_as_of"])
    latest = source["update_as_of"].max()
    source = source[source["update_as_of"] == latest].rename(
        columns={
            "long_name": "player",
            "player_positions": "position",
            "overall": "talent",
            "club_name": "club",
            "nationality_name": "team",
        }
    )
    return _select_squad(source, latest.date().isoformat())


def fc25(path: str | None) -> pd.DataFrame:
    with _archive(FC25_URL, path) as archive:
        source = pd.read_csv(
            archive.open("player-data-full-2025-june.csv"),
            usecols=[
                "full_name",
                "positions",
                "overall_rating",
                "dob",
                "club_name",
                "country_name",
            ],
        )
    as_of = pd.Timestamp("2025-06-03")
    born = pd.to_datetime(source["dob"], errors="coerce")
    source["age"] = (as_of - born).dt.days / 365.25
    source = source.rename(
        columns={
            "full_name": "player",
            "positions": "position",
            "overall_rating": "talent",
            "club_name": "club",
            "country_name": "team",
        }
    )
    return _select_squad(source, as_of.date().isoformat())


def main(
    fifa_15_22_zip: str | None = None,
    fc24_zip: str | None = None,
    fc25_zip: str | None = None,
) -> None:
    current = pd.read_csv(OUTPUT)
    current["as_of"] = pd.to_datetime(current["as_of"])
    actual = current[current["as_of"] >= pd.Timestamp("2026-01-01")].copy()
    history = [
        *fifa_15_22(fifa_15_22_zip),
        fc24(fc24_zip),
        fc25(fc25_zip),
    ]
    output = pd.concat([*history, actual], ignore_index=True)
    output["as_of"] = pd.to_datetime(output["as_of"])
    output = output.sort_values(["as_of", "team", "is_starter", "talent"])
    output.to_csv(OUTPUT, index=False, date_format="%Y-%m-%d")
    SOURCES.write_text(
        "FIFA 15-22 annual player databases:\n"
        f"{FIFA_15_22_URL}\n\n"
        "EA Sports FC 24 player database:\n"
        f"{FC24_URL}\n\n"
        "EA Sports FC 25 player database (June 2025):\n"
        f"{FC25_URL}\n\n"
        "The June 2026 rows are the actual squad snapshot documented in "
        "squads_match_report.txt.\n"
    )
    print(
        f"wrote {len(output):,} player snapshots for "
        f"{output['team'].nunique()} teams from "
        f"{output['as_of'].min().date()} through {output['as_of'].max().date()}"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
