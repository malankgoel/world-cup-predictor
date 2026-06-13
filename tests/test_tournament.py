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

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = ROOT / "data" / "input" / "schedule_2026.csv"


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
