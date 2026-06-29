from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from worldcup_predictor.data import load_schedule
from worldcup_predictor.features import FeatureBuilder
from worldcup_predictor.model import WorldCupModel
from worldcup_predictor.tournament import (
    PENALTY_SKILL_WEIGHT,
    STAGE_OUTPUT,
    TournamentSimulator,
    assign_third_place_teams,
)


def _mini_bracket():
    schedule = pd.DataFrame(
        [
            {"match_id": 1, "stage": "group", "group": "A", "date": "2026-06-11",
             "home_team": "Alpha", "away_team": "Bravo",
             "home_source": "", "away_source": "", "venue": "X", "venue_country": "Y",
             "neutral": True},
            {"match_id": 2, "stage": "group", "group": "B", "date": "2026-06-11",
             "home_team": "Charlie", "away_team": "Delta",
             "home_source": "", "away_source": "", "venue": "X", "venue_country": "Y",
             "neutral": True},
            {"match_id": 3, "stage": "round_of_16", "group": "", "date": "2026-06-20",
             "home_team": "", "away_team": "",
             "home_source": "1A", "away_source": "1B", "venue": "X",
             "venue_country": "Y", "neutral": True},
        ]
    )
    schedule["date"] = pd.to_datetime(schedule["date"])
    return schedule


def test_resolve_first_knockout_maps_slots_to_real_teams():
    from worldcup_predictor.cli import resolve_first_knockout_matchups

    schedule = _mini_bracket()
    results = pd.DataFrame(
        [
            {"date": "2026-06-11", "home_team": "Alpha", "away_team": "Bravo",
             "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup"},
            {"date": "2026-06-11", "home_team": "Charlie", "away_team": "Delta",
             "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup"},
        ]
    )
    results["date"] = pd.to_datetime(results["date"])
    fixtures, _ = resolve_first_knockout_matchups(schedule, results)
    assert len(fixtures) == 1
    _, _, home, away = fixtures[0]
    # 1A is Alpha (beat Bravo), 1B is Charlie (beat Delta).
    assert (home, away) == ("Alpha", "Charlie")


def test_resolve_first_knockout_raises_when_group_stage_incomplete():
    from worldcup_predictor.cli import resolve_first_knockout_matchups

    schedule = _mini_bracket()
    results = pd.DataFrame(
        [
            {"date": "2026-06-11", "home_team": "Alpha", "away_team": "Bravo",
             "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup"},
        ]
    )
    results["date"] = pd.to_datetime(results["date"])
    with pytest.raises(ValueError, match="not yet in results"):
        resolve_first_knockout_matchups(schedule, results)


def test_merge_shootout_winners_fills_only_blank_winners(tmp_path):
    from worldcup_predictor.cli import merge_shootout_winners

    results = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-04", "2026-07-05", "2026-07-06"]),
            "home_team": ["Alpha", "Charlie", "Echo"],
            "away_team": ["Bravo", "Delta", "Foxtrot"],
            "winner": ["", "Charlie", ""],  # middle game already decided manually
        }
    )
    shootouts = pd.DataFrame(
        {
            "date": ["2026-07-04", "2026-07-05"],
            "home_team": ["Alpha", "Charlie"],
            "away_team": ["Bravo", "Delta"],
            "winner": ["Bravo", "Delta"],  # would overwrite the manual one if buggy
        }
    )
    path = tmp_path / "shootouts.csv"
    shootouts.to_csv(path, index=False)
    merged = merge_shootout_winners(results, str(path))
    by_home = merged.set_index("home_team")["winner"]
    assert by_home["Alpha"] == "Bravo"      # blank winner filled from shootouts
    assert by_home["Charlie"] == "Charlie"  # manual winner preserved (not clobbered)
    assert by_home["Echo"] == ""            # no shootout -> stays blank


def test_merge_shootout_winners_is_a_noop_without_a_file():
    from worldcup_predictor.cli import merge_shootout_winners

    results = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-04"]),
            "home_team": ["Alpha"],
            "away_team": ["Bravo"],
            "winner": [""],
        }
    )
    pd.testing.assert_frame_equal(merge_shootout_winners(results, None), results)


def test_shootout_probability_shrinks_favorite_toward_coin_flip():
    # No edge -> exactly 50/50.
    assert TournamentSimulator._shootout_probability(0.0) == 0.5
    # A favorite keeps only PENALTY_SKILL_WEIGHT of its open-play edge.
    elo_diff = 400.0
    full_edge = 1.0 / (1.0 + np.exp(-elo_diff / 400.0))
    shrunk = TournamentSimulator._shootout_probability(elo_diff)
    assert 0.5 < shrunk < full_edge
    assert np.isclose(shrunk - 0.5, PENALTY_SKILL_WEIGHT * (full_edge - 0.5))


def test_team_shock_tilts_rates_and_is_neutral_when_shared():
    strong, weak = (0.3, 0.1), (-0.3, 0.1)
    home_rate, away_rate = TournamentSimulator._apply_team_shock(
        1.5, 1.5, strong, weak
    )
    assert home_rate > away_rate  # the stronger draw scores more, concedes less
    # A shared shock cancels: the matchup is unchanged up to the unbiasing term.
    same = (0.2, 0.1)
    h2, a2 = TournamentSimulator._apply_team_shock(1.5, 1.2, same, same)
    assert abs(h2 / a2 - 1.5 / 1.2) < 1e-9


def test_simulation_samples_from_model_joint_score_grid():
    class StubModel:
        def outcome_probabilities(self, home_rate, away_rate):
            matrix = np.zeros((3, 3))
            matrix[0, 1] = 1.0
            return 0.0, 0.0, 1.0, matrix

    simulator = TournamentSimulator.__new__(TournamentSimulator)
    simulator.model = StubModel()
    home, away = simulator._sample_score(1.3, 1.1, np.random.default_rng(1))
    assert (home, away) == (0, 1)

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "data" / "input" / "schedule_2026.csv"
SCHEDULE_2022 = ROOT / "data" / "input" / "schedule_2022.csv"


def test_third_place_assignment_uses_each_team_once():
    sources = [
        "3A/B/C/D/F",
        "3C/D/F/G/H",
        "3C/E/F/H/I",
        "3E/H/I/J/K",
        "3B/E/F/I/J",
        "3A/E/H/I/J",
        "3E/F/G/I/J",
        "3D/E/I/J/L",
    ]
    qualified = {group: f"Team {group}" for group in "ACEFGIJK"}
    assigned = assign_third_place_teams(sources, qualified)
    assert len(assigned) == 8
    assert len(set(assigned.values())) == 8


def test_third_place_assignment_uses_official_2026_table_for_current_projection():
    sources = [
        "3A/B/C/D/F",
        "3C/D/F/G/H",
        "3C/E/F/H/I",
        "3E/H/I/J/K",
        "3B/E/F/I/J",
        "3A/E/H/I/J",
        "3E/F/G/I/J",
        "3D/E/I/J/L",
    ]
    qualified = {
        "B": "Bosnia and Herzegovina",
        "D": "Paraguay",
        "E": "Ecuador",
        "F": "Sweden",
        "I": "Senegal",
        "J": "Algeria",
        "K": "DR Congo",
        "L": "Ghana",
    }
    assigned = assign_third_place_teams(sources, qualified)
    assert assigned["3C/E/F/H/I"] == "Ecuador"
    assert assigned["3E/F/G/I/J"] == "Algeria"
    assert assigned["3B/E/F/I/J"] == "Bosnia and Herzegovina"
    assert assigned["3A/B/C/D/F"] == "Paraguay"
    assert assigned["3A/E/H/I/J"] == "Senegal"
    assert assigned["3C/D/F/G/H"] == "Sweden"
    assert assigned["3D/E/I/J/L"] == "Ghana"
    assert assigned["3E/H/I/J/K"] == "DR Congo"


@pytest.mark.skipif(not SCHEDULE.exists(), reason="shipped schedule not present")
def test_simulate_produces_a_valid_probability_table(monkeypatch):
    # Constant, label-symmetric goal rates let the full bracket run without a
    # fitted model, so this exercises group standings, third-place selection,
    # the bracket, and aggregation end to end.
    monkeypatch.setattr(
        WorldCupModel,
        "_rates",
        lambda self, frame: (np.full(len(frame), 1.3), np.full(len(frame), 1.3)),
    )
    schedule = load_schedule(SCHEDULE)
    builder = FeatureBuilder(pd.DataFrame(), pd.DataFrame())
    empty_results = pd.DataFrame(
        columns=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "neutral",
            "winner",
        ]
    )
    simulator = TournamentSimulator(
        WorldCupModel(), builder, empty_results, schedule
    )
    table = simulator.simulate(runs=20, seed=1)

    assert len(table) == 48
    columns = [*STAGE_OUTPUT.values(), "champion"]
    values = table[columns].to_numpy()
    assert ((values >= 0.0) & (values <= 1.0)).all()
    # Exactly one champion and 32 round-of-32 teams per run.
    assert np.isclose(table["champion"].sum(), 1.0)
    assert np.isclose(table["reach_round_of_32"].sum(), 32.0)


@pytest.mark.skipif(not SCHEDULE_2022.exists(), reason="2022 schedule not present")
def test_simulator_handles_the_2022_32_team_format(monkeypatch):
    # The engine must derive the bracket from the schedule: 32 teams, a Round
    # of 16 (not 32), no third-place qualifiers, final = match 64.
    monkeypatch.setattr(
        WorldCupModel,
        "_rates",
        lambda self, frame: (np.full(len(frame), 1.3), np.full(len(frame), 1.3)),
    )
    schedule = load_schedule(SCHEDULE_2022)
    builder = FeatureBuilder(pd.DataFrame(), pd.DataFrame())
    empty_results = pd.DataFrame(
        columns=[
            "date", "home_team", "away_team", "home_score",
            "away_score", "tournament", "neutral", "winner",
        ]
    )
    simulator = TournamentSimulator(WorldCupModel(), builder, empty_results, schedule)
    assert simulator.final_match_id == 64
    assert "round_of_16" in simulator.stage_outputs
    assert "round_of_32" not in simulator.stage_outputs
    table = simulator.simulate(runs=20, seed=1)
    assert len(table) == 32
    assert "reach_round_of_32" not in table.columns
    assert np.isclose(table["champion"].sum(), 1.0)
    assert np.isclose(table["reach_round_of_16"].sum(), 16.0)
    assert np.isclose(table["reach_final"].sum(), 2.0)
