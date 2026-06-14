"""Write data/input/schedule_2022.csv in the project's schedule format.

The 2022 World Cup was a 32-team event: eight groups of four, top two into a
Round of 16, then quarter-finals, semi-finals, a third-place match, and the
final (64 matches). Group compositions and the knockout source map below are
the real 2022 bracket; group fixtures are a full round robin (their order does
not affect group standings). All matches are marked neutral (Qatar hosted; the
host-advantage map is 2026-specific, and Qatar's effect is negligible here).
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "input" / "schedule_2022.csv"

GROUPS = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}

# (match_id, stage, home_source, away_source) for the real 2022 bracket.
KNOCKOUT = [
    (49, "round_of_16", "1A", "2B"),
    (50, "round_of_16", "1C", "2D"),
    (51, "round_of_16", "1D", "2C"),
    (52, "round_of_16", "1B", "2A"),
    (53, "round_of_16", "1E", "2F"),
    (54, "round_of_16", "1G", "2H"),
    (55, "round_of_16", "1F", "2E"),
    (56, "round_of_16", "1H", "2G"),
    (57, "quarter_final", "W49", "W50"),
    (58, "quarter_final", "W53", "W54"),
    (59, "quarter_final", "W55", "W56"),
    (60, "quarter_final", "W51", "W52"),
    (61, "semi_final", "W57", "W58"),
    (62, "semi_final", "W59", "W60"),
    (63, "third_place", "L61", "L62"),
    (64, "final", "W61", "W62"),
]

COLUMNS = [
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
]


def _round_robin(teams: list[str]) -> list[tuple[str, str]]:
    a, b, c, d = teams
    return [(a, b), (c, d), (a, c), (b, d), (a, d), (b, c)]


def build_rows() -> list[dict]:
    rows = []
    match_id = 1
    start = date(2022, 11, 20)
    for group, teams in GROUPS.items():
        for home, away in _round_robin(teams):
            rows.append(
                {
                    "match_id": match_id,
                    "date": (start + timedelta(days=(match_id - 1) // 4)).isoformat(),
                    "stage": "group",
                    "group": group,
                    "home_team": home,
                    "away_team": away,
                    "home_source": home,
                    "away_source": away,
                    "venue_country": "Qatar",
                    "neutral": True,
                }
            )
            match_id += 1
    knockout_start = date(2022, 12, 3)
    for offset, (mid, stage, home_source, away_source) in enumerate(KNOCKOUT):
        rows.append(
            {
                "match_id": mid,
                "date": (knockout_start + timedelta(days=offset)).isoformat(),
                "stage": stage,
                "group": "",
                "home_team": "",
                "away_team": "",
                "home_source": home_source,
                "away_source": away_source,
                "venue_country": "Qatar",
                "neutral": True,
            }
        )
    return rows


def main() -> None:
    rows = build_rows()
    assert len(rows) == 64, f"expected 64 matches, built {len(rows)}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} matches to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
