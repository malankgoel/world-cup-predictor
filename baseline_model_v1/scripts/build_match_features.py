"""Build optional historical match features from official StatsBomb open data."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldcup_predictor.data import normalize_team  # noqa: E402

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
WORLD_CUP_SEASONS = (3, 106)  # 2018 and 2022
OUTPUT = ROOT / "data" / "input" / "match_features.csv"
SOURCES = ROOT / "data" / "input" / "match_features_sources.txt"


def get_json(session: requests.Session, path: str):
    response = session.get(f"{BASE}/{path}", timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    session = requests.Session()
    rows = []
    for season_id in WORLD_CUP_SEASONS:
        matches = get_json(session, f"matches/43/{season_id}.json")
        for index, match in enumerate(matches, 1):
            events = get_json(session, f"events/{match['match_id']}.json")
            xg = {}
            for event in events:
                if event.get("type", {}).get("name") != "Shot":
                    continue
                if int(event.get("period", 0)) > 4:
                    continue
                team = normalize_team(event["team"]["name"])
                value = float(event.get("shot", {}).get("statsbomb_xg", 0.0))
                xg[team] = xg.get(team, 0.0) + value
            home = normalize_team(match["home_team"]["home_team_name"])
            away = normalize_team(match["away_team"]["away_team_name"])
            rows.append(
                {
                    "date": match["match_date"],
                    "home_team": home,
                    "away_team": away,
                    "home_xg": xg.get(home, 0.0),
                    "away_xg": xg.get(away, 0.0),
                }
            )
            if index % 16 == 0:
                print(
                    f"season {season_id}: {index}/{len(matches)} matches",
                    flush=True,
                )
    output = pd.DataFrame(rows).sort_values(["date", "home_team"])
    if OUTPUT.exists() and OUTPUT.stat().st_size:
        existing = pd.read_csv(OUTPUT)
        other = [
            column
            for column in existing.columns
            if column not in {"home_xg", "away_xg"}
        ]
        if set(("date", "home_team", "away_team")) <= set(other):
            output = existing[other].merge(
                output,
                on=["date", "home_team", "away_team"],
                how="outer",
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    SOURCES.write_text(
        "StatsBomb Open Data (FIFA World Cup 2018 and 2022 event data):\n"
        "https://github.com/statsbomb/open-data\n"
    )
    print(f"wrote xG for {len(rows)} World Cup matches to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
