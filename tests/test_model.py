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


def test_blend_market_pulls_toward_a_sharper_market_favorite():
    model_probs = (0.55, 0.25, 0.20)  # mild home favorite
    market_probs = (0.80, 0.13, 0.07)  # market is far more confident
    blended = WorldCupModel.blend_market(model_probs, market_probs, weight=0.5)
    assert np.isclose(sum(blended), 1.0)
    # The blend must move the favorite up toward the market, fixing the
    # under-pricing, without overshooting the market itself.
    assert model_probs[0] < blended[0] < market_probs[0]


def test_blend_market_is_a_noop_without_usable_odds():
    model_probs = (0.55, 0.25, 0.20)
    # weight 0 disables it; missing/degenerate market vectors are ignored.
    assert WorldCupModel.blend_market(model_probs, (0.8, 0.1, 0.1), 0.0) == model_probs
    assert WorldCupModel.blend_market(
        model_probs, (np.nan, np.nan, np.nan), 0.5
    ) == model_probs
    assert WorldCupModel.blend_market(model_probs, (0.0, 0.0, 0.0), 0.5) == model_probs


def test_knockout_advance_probabilities_have_no_draw_and_favor_the_stronger_side():
    model = WorldCupModel()
    model.rho = -0.05
    home_adv, away_adv = model.knockout_advance_probabilities(
        1.8, 1.0, elo_diff=150.0
    )
    # A knockout must resolve: the two advance probabilities sum to 1 (no draw).
    assert np.isclose(home_adv + away_adv, 1.0)
    # The stronger side advances more often than its 90-minute win prob, because
    # half the draw mass (via extra time / a near-even shootout) also goes its way.
    home90, draw90, away90, _ = model.outcome_probabilities(1.8, 1.0)
    assert home_adv > home90
    assert home_adv > away_adv


def test_knockout_advance_uses_supplied_ninety_minute_split():
    model = WorldCupModel()
    # With an even 90-minute split and no Elo edge, advancing is a coin flip.
    home_adv, away_adv = model.knockout_advance_probabilities(
        1.3, 1.3, elo_diff=0.0, ninety=(0.4, 0.2, 0.4)
    )
    assert np.isclose(home_adv, 0.5, atol=1e-6)
    assert np.isclose(away_adv, 0.5, atol=1e-6)


def test_monotonic_constraints_are_attached_to_the_goal_model():
    model = WorldCupModel()
    constraints = model.goal_model.get_params()["monotonic_cst"]
    assert constraints["home_elo"] == 1
    assert constraints["away_elo"] == -1
    # Disabling the flag must drop the constraints entirely.
    assert WorldCupModel(monotonic=False).goal_model.get_params()[
        "monotonic_cst"
    ] is None


def test_importance_power_sharpens_tournament_weighting():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "importance": [1.0, 4.0],
        }
    )
    low = WorldCupModel(importance_power=0.5)._sample_weights(frame, 4.0)
    high = WorldCupModel(importance_power=1.0)._sample_weights(frame, 4.0)
    # A higher power widens the gap between a friendly and a major match.
    assert high.iloc[1] / high.iloc[0] > low.iloc[1] / low.iloc[0]
