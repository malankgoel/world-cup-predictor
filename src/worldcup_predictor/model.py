from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, (os.cpu_count() or 2) - 1))

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error

from .features import FEATURE_COLUMNS

# Hand-set multiplicative priors applied in `_rates` ONLY for signals that were
# all-NaN during training and therefore never learned by the goal models (e.g.
# squad talent or FIFA ranking points, when only a current snapshot exists).
# These coefficients are deliberate priors, NOT learned or cross-validated, so
# they are centralized here to be easy to find, tune, and document. Supply
# historical snapshots instead if you want the effect learned and validated.
DEFAULT_PRIOR_ADJUSTMENTS = {
    "squad_coef": 0.025,
    "squad_clip": 0.25,
    "rank_coef": 0.001,
    "rank_clip": 0.25,
    "chemistry_coef": 0.08,
    "chemistry_clip": 0.5,
}


def flip_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the same matches with the home/away labels swapped."""
    flipped = {}
    for column in frame.columns:
        if column.startswith("home_"):
            flipped[column] = frame["away_" + column[5:]]
        elif column.startswith("away_"):
            flipped[column] = frame["home_" + column[5:]]
        elif column.endswith("_diff"):
            flipped[column] = -frame[column]
        else:
            flipped[column] = frame[column]
    return pd.DataFrame(flipped, index=frame.index)


@dataclass
class MatchPrediction:
    expected_home_goals: float
    expected_away_goals: float
    home_win: float
    draw: float
    away_win: float
    most_likely_score: str
    over_2_5: float
    under_2_5: float
    both_teams_to_score: float


class WorldCupModel:
    def __init__(
        self,
        parameters: dict | None = None,
        adjustments: dict | None = None,
    ):
        parameters = parameters or {}
        defaults = {
            "loss": "poisson",
            "learning_rate": 0.05,
            "max_iter": 250,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "random_state": 42,
        }
        defaults.update(parameters)
        self.parameters = defaults
        self.adjustments = {**DEFAULT_PRIOR_ADJUSTMENTS, **(adjustments or {})}
        self.rho = 0.0
        self.elo_baseline: LogisticRegression | None = None
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.home_model = self._new_model()
        self.away_model = self._new_model()

    def _new_model(self):
        return HistGradientBoostingRegressor(**self.parameters)

    @staticmethod
    def score_matrix(
        home_rate: float,
        away_rate: float,
        max_goals: int = 10,
        rho: float = 0.0,
    ):
        goals = np.arange(max_goals + 1)
        matrix = np.outer(poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate))
        matrix[0, 0] *= 1 - home_rate * away_rate * rho
        matrix[0, 1] *= 1 + home_rate * rho
        matrix[1, 0] *= 1 + away_rate * rho
        matrix[1, 1] *= 1 - rho
        matrix = np.clip(matrix, 0, None)
        return matrix / matrix.sum()

    def outcome_probabilities(self, home_rate: float, away_rate: float):
        matrix = self.score_matrix(home_rate, away_rate, rho=self.rho)
        home_win = float(np.tril(matrix, -1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, 1).sum())
        return home_win, draw, away_win, matrix

    def _fit_rho(
        self,
        frame: pd.DataFrame,
        home_rates: np.ndarray,
        away_rates: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        home_goals = frame["home_score"].to_numpy()
        away_goals = frame["away_score"].to_numpy()

        def objective(rho):
            tau = np.ones(len(frame))
            tau[(home_goals == 0) & (away_goals == 0)] = (
                1 - home_rates * away_rates * rho
            )[(home_goals == 0) & (away_goals == 0)]
            tau[(home_goals == 0) & (away_goals == 1)] = (
                1 + home_rates * rho
            )[(home_goals == 0) & (away_goals == 1)]
            tau[(home_goals == 1) & (away_goals == 0)] = (
                1 + away_rates * rho
            )[(home_goals == 1) & (away_goals == 0)]
            tau[(home_goals == 1) & (away_goals == 1)] = 1 - rho
            if np.any(tau <= 0):
                return np.inf
            return float(-np.sum(weights * np.log(tau)))

        return float(
            minimize_scalar(objective, bounds=(-0.20, 0.20), method="bounded").x
        )

    @staticmethod
    def _elo_baseline_features(frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                frame["elo_diff"].to_numpy(float),
                (frame["home_advantage"] - frame["away_advantage"]).to_numpy(float),
            ]
        )

    @staticmethod
    def _outcome_classes(frame: pd.DataFrame) -> np.ndarray:
        return np.where(
            frame["home_score"] > frame["away_score"],
            0,
            np.where(frame["home_score"] == frame["away_score"], 1, 2),
        )

    def _rates(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # Two separately fitted goal models make predictions depend on which
        # team carries the arbitrary "home" label, which matters on neutral
        # venues; averaging both orientations removes the label dependence
        # (venue advantage is preserved because flipping swaps it too).
        flipped = flip_features(features)
        home = 0.5 * (
            self.home_model.predict(features[self.feature_columns])
            + self.away_model.predict(flipped[self.feature_columns])
        )
        away = 0.5 * (
            self.away_model.predict(features[self.feature_columns])
            + self.home_model.predict(flipped[self.feature_columns])
        )
        adjustments = getattr(self, "adjustments", DEFAULT_PRIOR_ADJUSTMENTS)
        if "home_squad_attack" not in self.feature_columns:
            coef, clip = adjustments["squad_coef"], adjustments["squad_clip"]
            home_edge = (
                features["home_squad_attack"] - features["away_squad_defense"]
            ).fillna(0)
            away_edge = (
                features["away_squad_attack"] - features["home_squad_defense"]
            ).fillna(0)
            home *= np.exp(np.clip(coef * home_edge, -clip, clip))
            away *= np.exp(np.clip(coef * away_edge, -clip, clip))
        if "home_rank_points" not in self.feature_columns:
            coef, clip = adjustments["rank_coef"], adjustments["rank_clip"]
            rank_edge = features["rank_points_diff"].fillna(0)
            rank_adjustment = np.clip(coef * rank_edge, -clip, clip)
            home *= np.exp(rank_adjustment)
            away *= np.exp(-rank_adjustment)
        if "home_chemistry" not in self.feature_columns:
            coef, clip = adjustments["chemistry_coef"], adjustments["chemistry_clip"]
            chemistry_edge = features["chemistry_diff"].fillna(0).clip(-clip, clip)
            home *= np.exp(coef * chemistry_edge)
            away *= np.exp(-coef * chemistry_edge)
        home = np.clip(home, 0.05, 6.0)
        away = np.clip(away, 0.05, 6.0)
        return home, away

    @staticmethod
    def _sample_weights(frame: pd.DataFrame, half_life_years: float) -> pd.Series:
        """Exponential recency decay times sqrt(match importance)."""
        age_days = (frame["date"].max() - frame["date"]).dt.days
        weights = np.exp(-np.log(2) * age_days / (365.25 * half_life_years))
        return weights * np.sqrt(frame["importance"].clip(lower=0.1))

    def fit_window(
        self, frame: pd.DataFrame, half_life_years: float = 4.0
    ) -> WorldCupModel:
        """Fit goal models, Dixon-Coles rho, and the Elo baseline on `frame` only.

        Unlike `fit`, this does no internal validation split and no full-data
        refit; it is the train-on-this-slice primitive used by the expanding
        window backtest.
        """
        frame = frame.sort_values("date").reset_index(drop=True)
        self.feature_columns = [
            column for column in FEATURE_COLUMNS if frame[column].notna().any()
        ]
        weights = self._sample_weights(frame, half_life_years)
        self.home_model = self._new_model()
        self.away_model = self._new_model()
        self.home_model.fit(
            frame[self.feature_columns], frame["home_score"], sample_weight=weights
        )
        self.away_model.fit(
            frame[self.feature_columns], frame["away_score"], sample_weight=weights
        )
        home_rates, away_rates = self._rates(frame)
        self.rho = self._fit_rho(frame, home_rates, away_rates, weights.to_numpy())
        self.elo_baseline = LogisticRegression(max_iter=1000)
        self.elo_baseline.fit(
            self._elo_baseline_features(frame),
            self._outcome_classes(frame),
            sample_weight=weights.to_numpy(),
        )
        return self

    def fit(
        self,
        frame: pd.DataFrame,
        validation_fraction: float = 0.2,
        half_life_years: float = 4.0,
    ) -> dict[str, float]:
        if len(frame) < 500:
            raise ValueError("At least 500 historical matches are required")
        frame = frame.sort_values("date").reset_index(drop=True)
        split = int(len(frame) * (1.0 - validation_fraction))
        train, valid = frame.iloc[:split], frame.iloc[split:]
        self.feature_columns = [
            column for column in FEATURE_COLUMNS if train[column].notna().any()
        ]
        weights = self._sample_weights(train, half_life_years)
        self.home_model.fit(
            train[self.feature_columns], train["home_score"], sample_weight=weights
        )
        self.away_model.fit(
            train[self.feature_columns], train["away_score"], sample_weight=weights
        )
        train_home_rates, train_away_rates = self._rates(train)
        self.rho = self._fit_rho(
            train, train_home_rates, train_away_rates, weights.to_numpy()
        )
        # A fair, learned 1X2 baseline: multinomial logistic on the Elo gap and
        # venue advantage, trained on the same recency/importance weights. This
        # replaces the old fixed-2.5-goal Elo->rate mapping.
        self.elo_baseline = LogisticRegression(max_iter=1000)
        self.elo_baseline.fit(
            self._elo_baseline_features(train),
            self._outcome_classes(train),
            sample_weight=weights.to_numpy(),
        )
        metrics = self.evaluate(valid)

        full_weights = self._sample_weights(frame, half_life_years)
        self.home_model = self._new_model()
        self.away_model = self._new_model()
        self.home_model.fit(
            frame[self.feature_columns], frame["home_score"], sample_weight=full_weights
        )
        self.away_model.fit(
            frame[self.feature_columns], frame["away_score"], sample_weight=full_weights
        )
        full_home_rates, full_away_rates = self._rates(frame)
        self.rho = self._fit_rho(
            frame, full_home_rates, full_away_rates, full_weights.to_numpy()
        )
        metrics["training_matches"] = int(len(frame))
        metrics["validation_matches"] = int(len(valid))
        metrics["training_through"] = frame["date"].max().date().isoformat()
        metrics["dixon_coles_rho"] = self.rho
        metrics["active_features"] = self.feature_columns
        return metrics

    @staticmethod
    def _binary_calibration_error(prob: np.ndarray, actual: np.ndarray) -> float:
        """Frequency-weighted gap between predicted and observed rates (10 bins)."""
        error = 0.0
        for low in np.linspace(0, 0.9, 10):
            mask = (prob >= low) & (prob < low + 0.1)
            if mask.any():
                error += mask.mean() * abs(
                    float(actual[mask].mean()) - float(prob[mask].mean())
                )
        return float(error)

    def _elo_baseline_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        if self.elo_baseline is not None:
            return self.elo_baseline.predict_proba(
                self._elo_baseline_features(frame)
            )
        # Fallback when no baseline was fitted (e.g. evaluate called standalone):
        # the old fixed-2.5-goal Elo->rate mapping.
        rows = []
        for row in frame.itertuples():
            adjusted_diff = row.elo_diff + 80 * (
                row.home_advantage - row.away_advantage
            )
            ratio = np.exp(adjusted_diff / 400)
            elo_home_rate = 2.5 * ratio / (1 + ratio)
            rows.append(self.outcome_probabilities(elo_home_rate, 2.5 - elo_home_rate)[:3])
        return np.asarray(rows)

    @staticmethod
    def _rps(probabilities: np.ndarray, one_hot: np.ndarray) -> float:
        return float(
            np.mean(
                np.sum(
                    (
                        np.cumsum(probabilities, axis=1)[:, :2]
                        - np.cumsum(one_hot, axis=1)[:, :2]
                    )
                    ** 2,
                    axis=1,
                )
                / 2
            )
        )

    def evaluate(self, frame: pd.DataFrame) -> dict[str, float]:
        home_rates, away_rates = self._rates(frame)
        probabilities = []
        over_2_5 = []
        both_teams_to_score = []
        for home_rate, away_rate in zip(home_rates, away_rates, strict=False):
            home_win, draw, away_win, matrix = self.outcome_probabilities(
                home_rate, away_rate
            )
            probabilities.append((home_win, draw, away_win))
            goals = np.add.outer(
                np.arange(matrix.shape[0]), np.arange(matrix.shape[1])
            )
            over_2_5.append(float(matrix[goals >= 3].sum()))
            both_teams_to_score.append(float(matrix[1:, 1:].sum()))
        probabilities = np.asarray(probabilities)
        over_2_5 = np.asarray(over_2_5)
        both_teams_to_score = np.asarray(both_teams_to_score)
        outcome = self._outcome_classes(frame)
        one_hot = np.eye(3)[outcome]
        confidence = probabilities.max(axis=1)
        correct = probabilities.argmax(axis=1) == outcome
        calibration_error = self._binary_calibration_error(confidence, correct)
        rps = self._rps(probabilities, one_hot)

        total_goals = (frame["home_score"] + frame["away_score"]).to_numpy()
        actual_over = (total_goals >= 3).astype(float)
        actual_btts = (
            (frame["home_score"] >= 1) & (frame["away_score"] >= 1)
        ).to_numpy().astype(float)

        elo_probabilities = self._elo_baseline_probabilities(frame)
        return {
            "outcome_log_loss": float(
                log_loss(outcome, probabilities, labels=[0, 1, 2])
            ),
            "ranked_probability_score": float(rps),
            "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            "calibration_error": float(calibration_error),
            "over_2_5_brier": float(np.mean((over_2_5 - actual_over) ** 2)),
            "over_2_5_calibration_error": self._binary_calibration_error(
                over_2_5, actual_over
            ),
            "both_teams_to_score_brier": float(
                np.mean((both_teams_to_score - actual_btts) ** 2)
            ),
            "both_teams_to_score_calibration_error": self._binary_calibration_error(
                both_teams_to_score, actual_btts
            ),
            "elo_baseline_log_loss": float(
                log_loss(outcome, elo_probabilities, labels=[0, 1, 2])
            ),
            "elo_baseline_rps": self._rps(elo_probabilities, one_hot),
            "home_goal_mae": float(
                mean_absolute_error(frame["home_score"], home_rates)
            ),
            "away_goal_mae": float(
                mean_absolute_error(frame["away_score"], away_rates)
            ),
        }

    def predict(self, features: pd.Series | dict) -> MatchPrediction:
        frame = pd.DataFrame([features])
        home_rate, away_rate = (values[0] for values in self._rates(frame))
        home_win, draw, away_win, matrix = self.outcome_probabilities(
            home_rate, away_rate
        )
        home_goals, away_goals = np.unravel_index(np.argmax(matrix), matrix.shape)
        over_2_5 = float(
            sum(
                matrix[home_goals, away_goals]
                for home_goals in range(matrix.shape[0])
                for away_goals in range(matrix.shape[1])
                if home_goals + away_goals >= 3
            )
        )
        both_score = float(matrix[1:, 1:].sum())
        return MatchPrediction(
            float(home_rate),
            float(away_rate),
            home_win,
            draw,
            away_win,
            # Spaces around the hyphen stop spreadsheets coercing e.g. "2-0"
            # into a date when the CSV is opened.
            f"{home_goals} - {away_goals}",
            over_2_5,
            1 - over_2_5,
            both_score,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> WorldCupModel:
        return joblib.load(path)
