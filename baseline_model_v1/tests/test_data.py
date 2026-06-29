from pathlib import Path

import pytest

from worldcup_predictor.data import _parse_schedule, load_schedule, normalize_team

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "data" / "input" / "schedule_2026.csv"


def test_normalize_team_applies_aliases_and_strips_whitespace():
    assert normalize_team("USA") == "United States"
    assert normalize_team("Korea Republic") == "South Korea"
    assert normalize_team("Türkiye") == "Turkey"
    assert normalize_team("Côte d'Ivoire") == "Ivory Coast"
    assert normalize_team("  Brazil  ") == "Brazil"


def test_parse_schedule_accepts_completed_and_future_fixtures():
    text = """\
▪ Group A
Thu June 11
  13:00 UTC-6 Mexico 2-0 (1-0) South Africa @ Mexico City
Thu June 18
  12:00 UTC-4 Czech Republic v South Africa @ Atlanta
"""

    rows = _parse_schedule(text, knockout=False)

    assert len(rows) == 2
    assert rows[0]["home_team"] == "Mexico"
    assert rows[0]["away_team"] == "South Africa"
    assert rows[1]["home_team"] == "Czech Republic"
    assert rows[1]["away_team"] == "South Africa"


@pytest.mark.skipif(not SCHEDULE.exists(), reason="shipped schedule not present")
def test_shipped_schedule_has_104_matches_with_expected_stage_split():
    schedule = load_schedule(SCHEDULE)
    assert len(schedule) == 104
    counts = schedule["stage"].value_counts().to_dict()
    assert counts["group"] == 72
    knockout = sum(value for stage, value in counts.items() if stage != "group")
    assert knockout == 32
    assert schedule["match_id"].is_unique
    # The final must be the last match so the simulator can read winners[104].
    assert int(schedule["match_id"].max()) == 104
    assert schedule.loc[schedule["match_id"] == 104, "stage"].iloc[0] == "final"
