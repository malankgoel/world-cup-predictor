import numpy as np
import pandas as pd

from worldcup_predictor.backtest import expanding_window_backtest, summarize_backtest
from worldcup_predictor.features import FeatureBuilder
from worldcup_predictor.model import WorldCupModel


def _synthetic_results(n_matches: int = 800, n_teams: int = 24, seed: int = 0):
    """A learnable synthetic history: per-team latent strength drives scores."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    strength = {team: rng.normal(0.0, 0.4) for team in teams}
    start = pd.Timestamp("2016-01-01")
    rows = []
    for k in range(n_matches):
        home, away = rng.choice(teams, size=2, replace=False)
        home_rate = np.exp(0.2 + strength[home] - strength[away])
        away_rate = np.exp(strength[away] - strength[home])
        rows.append(
            {
                "date": start + pd.Timedelta(days=int(k * 3)),
                "home_team": home,
                "away_team": away,
                "home_score": int(rng.poisson(home_rate)),
                "away_score": int(rng.poisson(away_rate)),
                "tournament": "Friendly" if k % 3 else "FIFA World Cup qualification",
                "neutral": False,
                "country": home,
            }
        )
    return pd.DataFrame(rows)


def _feature_frame():
    return FeatureBuilder(pd.DataFrame(), pd.DataFrame()).training_frame(
        _synthetic_results()
    )


def test_fit_reports_secondary_market_and_baseline_metrics():
    model = WorldCupModel({"max_iter": 60})
    metrics = model.fit(_feature_frame(), validation_fraction=0.25)
    expected = [
        "ranked_probability_score",
        "over_2_5_brier",
        "over_2_5_calibration_error",
        "both_teams_to_score_brier",
        "both_teams_to_score_calibration_error",
        "elo_baseline_log_loss",
        "elo_baseline_rps",
    ]
    for key in expected:
        assert key in metrics and np.isfinite(metrics[key])
    assert 0.0 <= metrics["over_2_5_brier"] <= 1.0
    assert 0.0 <= metrics["both_teams_to_score_brier"] <= 1.0
    assert model.elo_baseline is not None


def test_expanding_window_backtest_runs_multiple_folds():
    table = expanding_window_backtest(
        _feature_frame(),
        cutoffs=["2020-01-01", "2021-01-01", "2022-01-01"],
        model_parameters={"max_iter": 60},
        min_train=100,
    )
    assert len(table) >= 2
    assert {"fold_start", "train_matches", "test_matches"} <= set(table.columns)
    assert "ranked_probability_score" in table.columns
    summary = summarize_backtest(table)
    assert np.isfinite(summary["ranked_probability_score"])
    # Match counts are not metrics, so they stay out of the averaged summary.
    assert "train_matches" not in summary
