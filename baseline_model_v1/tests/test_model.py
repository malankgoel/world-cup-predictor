import numpy as np
import pandas as pd

from worldcup_predictor.features import FEATURE_COLUMNS
from worldcup_predictor.model import WorldCupModel, flip_features


def test_outcome_probabilities_sum_to_one():
    model = WorldCupModel()
    model.rho = -0.08
    home, draw, away, matrix = model.outcome_probabilities(1.7, 1.1)
    assert np.isclose(home + draw + away, 1)
    assert np.isclose(matrix.sum(), 1)
    assert home > away


def test_dixon_coles_changes_low_score_cells():
    plain = WorldCupModel.score_matrix(1.2, 1.0, rho=0)
    corrected = WorldCupModel.score_matrix(1.2, 1.0, rho=-0.08)
    assert not np.isclose(plain[0, 0], corrected[0, 0])


def test_flip_features_is_an_involution_and_swaps_sides():
    frame = pd.DataFrame([dict.fromkeys(FEATURE_COLUMNS, 0.0)])
    frame["home_elo"] = 1900.0
    frame["away_elo"] = 1700.0
    frame["elo_diff"] = 200.0
    frame["home_advantage"] = 1.0
    flipped = flip_features(frame)
    assert flipped.loc[0, "home_elo"] == 1700.0
    assert flipped.loc[0, "away_advantage"] == 1.0
    assert flipped.loc[0, "elo_diff"] == -200.0
    pd.testing.assert_frame_equal(flip_features(flipped), frame)


def test_prior_adjustment_coefficients_are_configurable():
    base = WorldCupModel()
    custom = WorldCupModel(adjustments={"chemistry_coef": 0.0})
    assert base.adjustments["chemistry_coef"] == 0.0
    assert custom.adjustments["chemistry_coef"] == 0.0

    class Stub:
        def predict(self, frame):
            return np.full(len(frame), 1.2)

    frame = pd.DataFrame([dict.fromkeys(FEATURE_COLUMNS, 0.0)])
    frame["chemistry_diff"] = 0.4
    rates = {}
    for name, model in (("base", base), ("custom", custom)):
        model.goal_model = Stub()
        # Drop chemistry from the learned set so the prior-nudge path runs.
        model.learned_features = set(FEATURE_COLUMNS) - {"home_chemistry"}
        rates[name] = float(model._rates(frame)[0][0])
    # The default is now neutral because historical snapshots let the model
    # learn this signal instead of relying on an unvalidated manual nudge.
    assert np.isclose(rates["base"], rates["custom"])
    assert np.isclose(rates["custom"], 1.2)


def test_rates_do_not_depend_on_home_away_label():
    class Stub:
        def __init__(self, intercept, slope):
            self.intercept = intercept
            self.slope = slope

        def predict(self, frame):
            return self.intercept + self.slope * frame["elo_diff"].to_numpy()

    model = WorldCupModel()
    model.goal_model = Stub(1.4, 0.002)
    model.learned_features = set(FEATURE_COLUMNS)
    matchup = pd.DataFrame([dict.fromkeys(FEATURE_COLUMNS, 0.0)])
    matchup["home_elo"], matchup["away_elo"] = 1900.0, 1700.0
    matchup["elo_diff"] = 200.0
    mirrored = flip_features(matchup)
    home_rate, away_rate = model._rates(matchup)
    mirrored_home, mirrored_away = model._rates(mirrored)
    assert np.isclose(home_rate[0], mirrored_away[0])
    assert np.isclose(away_rate[0], mirrored_home[0])


def test_negative_binomial_dispersion_increases_tail_probability():
    poisson_grid = WorldCupModel.score_matrix(1.3, 1.1, dispersion=0.0)
    overdispersed = WorldCupModel.score_matrix(1.3, 1.1, dispersion=0.25)
    goals = np.add.outer(
        np.arange(poisson_grid.shape[0]),
        np.arange(poisson_grid.shape[1]),
    )
    assert overdispersed[goals >= 5].sum() > poisson_grid[goals >= 5].sum()


def test_temperature_above_one_flattens_outcome_probabilities():
    sharp = WorldCupModel()
    sharp.temperature = 1.0
    tempered = WorldCupModel()
    tempered.temperature = 2.0
    sharp_probs = sharp.outcome_probabilities(1.9, 0.9)[:3]
    tempered_probs = tempered.outcome_probabilities(1.9, 0.9)[:3]
    # A temperature above 1 shrinks the favourite's probability toward uniform.
    assert max(tempered_probs) < max(sharp_probs)
    assert np.isclose(sum(tempered_probs), 1.0)


def test_fixed_temperature_is_stored_for_fit_window():
    model = WorldCupModel(calibration_temperature=0.9)
    assert model.fixed_temperature == 0.9
    assert WorldCupModel().fixed_temperature is None
