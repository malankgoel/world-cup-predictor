from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from worldcup_predictor.data import load_schedule
from worldcup_predictor.features import FeatureBuilder
from worldcup_predictor.model import WorldCupModel
from worldcup_predictor.tournament import (
    STAGE_OUTPUT,
    TournamentSimulator,
    assign_third_place_teams,
)

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
