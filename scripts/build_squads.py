"""Build data/input/squads.csv from Wikipedia squad lists and EA FC ratings.

Inputs:
  /tmp/wc_squads.html        saved copy of en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads
  data/raw/ea_ratings.json   produced by scripts/download_ea_ratings.py

Outputs:
  data/input/squads.csv
  data/input/squads_match_report.txt

Matching is restricted to the player's national team (EA nationality), in order:
exact normalized name, EA common name, same birthdate plus shared name token,
then fuzzy name ratio >= 0.85 (recorded with a higher talent_se).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from datetime import date, datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worldcup_predictor.data import load_schedule, normalize_team  # noqa: E402

AS_OF = "2026-06-01"
POSITION_MAP = {"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD"}

# Schedule team name -> EA nationality label, where they differ.
EA_NATIONALITY = {
    "Cape Verde": "Cape Verde Islands",
    "DR Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
    "Netherlands": "Holland",
    "South Korea": "Korea Republic",
}

# Players absent from the EA database fall into two groups. On teams EA covers
# well, a miss usually means an unlicensed club (Flamengo, Russian league,
# Liga MX), so the player is imputed near his squad's median. On thinly covered
# teams a miss usually means a weaker domestic league, so a conservative floor
# distorts less than dropping the row, which would leave small federations
# represented only by their few Europe-based stars.
IMPUTED_FLOOR = 62.0
IMPUTED_SE = 8.0


def imputed_talent(rated_talents: list[float]) -> float:
    """Squad median minus a discount that widens as coverage shrinks.

    Well-covered squads (13+ matched) get median - 5; each missing player below
    that adds a point, so thinly covered teams converge to the floor.
    """
    if len(rated_talents) < 2:
        return IMPUTED_FLOOR
    median = float(pd.Series(rated_talents).median())
    discount = 5.0 + max(0, 13 - len(rated_talents))
    return float(min(76.0, max(IMPUTED_FLOOR, median - discount)))


def fold(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("-", " ").replace("'", "").replace(".", "")
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_squads(html_path: str, teams: set[str]) -> pd.DataFrame:
    soup = BeautifulSoup(Path(html_path).read_text(), "lxml")
    rows = []
    for heading in soup.find_all("h3"):
        team = normalize_team(heading.get_text(strip=True))
        if team not in teams:
            continue
        table = heading.find_next("table", class_="sortable")
        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["th", "td"])
            if len(cells) < 7:
                continue
            text = [cell.get_text(" ", strip=True) for cell in cells]
            position = POSITION_MAP.get(text[1].split()[-1])
            player = re.sub(r"\s*\(.*?\)\s*", " ", text[2]).strip()
            born = re.search(r"\(\s*(\d{4}-\d{2}-\d{2})\s*\)", text[3])
            caps = re.sub(r"\D", "", text[4])
            rows.append(
                {
                    "team": team,
                    "player": player,
                    "position": position,
                    "birthdate": born.group(1) if born else "",
                    "caps": int(caps) if caps else 0,
                    "club": text[6],
                }
            )
    return pd.DataFrame(rows)


def load_ea(path: str) -> pd.DataFrame:
    frame = pd.DataFrame(json.loads(Path(path).read_text()))
    frame = frame[frame["gender"] == 0].copy()
    frame["full"] = (frame["firstName"] + " " + frame["lastName"]).map(fold)
    frame["common"] = frame["commonName"].map(fold)
    # The API mixes "6/15/1992 12:00:00 AM" and "07/08/1991 00:00" formats.
    born = pd.to_datetime(
        frame["birthdate"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    fallback = pd.to_datetime(
        frame["birthdate"], format="%m/%d/%Y %H:%M", errors="coerce"
    )
    frame["born"] = born.fillna(fallback).dt.date.astype(str)
    return frame


def match_player(
    row: pd.Series, pool: pd.DataFrame, ea: pd.DataFrame
) -> tuple[float | None, float | None, str]:
    name = fold(row["player"])
    exact = pool[(pool["full"] == name) | (pool["common"] == name)]
    if len(exact):
        return float(exact["overallRating"].max()), 3.0, "exact"
    tokens = set(name.split())
    if len(tokens) >= 2:
        subset = pool[
            pool.apply(
                lambda c: tokens <= set((c["full"] + " " + c["common"]).split()),
                axis=1,
            )
        ]
        if len(subset):
            return float(subset["overallRating"].max()), 4.0, "tokens"
    if row["birthdate"]:
        same_birth = pool[pool["born"] == row["birthdate"]]
        for _, candidate in same_birth.iterrows():
            if tokens & set((candidate["full"] + " " + candidate["common"]).split()):
                return float(candidate["overallRating"]), 3.0, "birthdate"
    best_score, best_rating = 0.0, None
    for _, candidate in pool.iterrows():
        for ea_name in (candidate["full"], candidate["common"]):
            if not ea_name:
                continue
            score = SequenceMatcher(None, name, ea_name).ratio()
            if score > best_score:
                best_score, best_rating = score, float(candidate["overallRating"])
    if best_score >= 0.85:
        return best_rating, 5.0, f"fuzzy({best_score:.2f})"
    # EA's nationality can lag national-team switches (dual nationals), so fall
    # back to the full database with stricter rules: a unique exact name, or a
    # birthdate plus a shared name token.
    global_exact = ea[(ea["full"] == name) | (ea["common"] == name)]
    if len(global_exact) == 1:
        return float(global_exact["overallRating"].iloc[0]), 4.0, "global-exact"
    if row["birthdate"]:
        same_birth = ea[ea["born"] == row["birthdate"]]
        for _, candidate in same_birth.iterrows():
            if tokens & set((candidate["full"] + " " + candidate["common"]).split()):
                return float(candidate["overallRating"]), 4.0, "global-birthdate"
    return None, None, "unmatched"


def main() -> None:
    schedule = load_schedule("data/input/schedule_2026.csv")
    group = schedule[schedule["stage"] == "group"]
    teams = set(group["home_team"]) | set(group["away_team"])
    squads = parse_squads("/tmp/wc_squads.html", teams)
    ea = load_ea("data/raw/ea_ratings.json")

    missing_teams = teams - set(squads["team"])
    if missing_teams:
        raise SystemExit(f"No squad table parsed for: {sorted(missing_teams)}")

    as_of = date.fromisoformat(AS_OF)
    output_rows, report = [], []
    for team, members in squads.groupby("team"):
        nationality = EA_NATIONALITY.get(team, team)
        pool = ea[ea["nationality"] == nationality]
        if pool.empty:
            report.append(f"!! {team}: no EA players under nationality {nationality!r}")
        matches = [match_player(row, pool, ea) for _, row in members.iterrows()]
        fill = imputed_talent(
            [talent for talent, _, how in matches if how != "unmatched"]
        )
        matched = []
        for (_, row), (talent, talent_se, how) in zip(
            members.iterrows(), matches, strict=False
        ):
            if how == "unmatched":
                talent, talent_se = fill, IMPUTED_SE
                report.append(
                    f"{team}: IMPUTED {fill:.0f} for {row['player']} "
                    f"({row['position']}, {row['club']})"
                )
            elif how != "exact":
                report.append(f"{team}: {how} {row['player']}")
            born = (
                datetime.strptime(row["birthdate"], "%Y-%m-%d").date()
                if row["birthdate"]
                else None
            )
            age = (
                round((as_of - born).days / 365.25, 1) if born else ""
            )
            matched.append(
                {
                    "as_of": AS_OF,
                    "team": team,
                    "player": row["player"],
                    "club": row["club"],
                    "position": row["position"],
                    "talent": talent if talent is not None else "",
                    "talent_se": talent_se if talent_se is not None else "",
                    "exp_minutes": "",
                    "available": True,
                    "age": age,
                    "is_starter": False,
                    "_caps": row["caps"],
                }
            )
        frame = pd.DataFrame(matched)
        rated = frame[frame["talent"] != ""].copy()
        rated["talent"] = rated["talent"].astype(float)
        keepers = rated[rated["position"] == "GK"].sort_values(
            ["talent", "_caps"], ascending=False
        )
        outfield = rated[rated["position"] != "GK"].sort_values(
            ["talent", "_caps"], ascending=False
        )
        starters = list(keepers.head(1).index) + list(outfield.head(10).index)
        frame.loc[starters, "is_starter"] = True
        output_rows.append(frame.drop(columns="_caps"))

    output = pd.concat(output_rows, ignore_index=True)
    output.to_csv("data/input/squads.csv", index=False)
    imputed = sum(1 for line in report if "IMPUTED" in line)
    matched_count = len(output) - imputed
    summary = (
        f"{len(output)} players across {output['team'].nunique()} teams; "
        f"{matched_count} matched to EA ratings ({matched_count / len(output):.1%}), "
        f"{imputed} imputed"
    )
    Path("data/input/squads_match_report.txt").write_text(
        summary + "\n\n" + "\n".join(report) + "\n"
    )
    print(summary)
    print(f"{len(report)} report lines -> data/input/squads_match_report.txt")


if __name__ == "__main__":
    main()
