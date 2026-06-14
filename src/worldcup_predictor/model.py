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
from scipy.stats import nbinom, poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error

from .features import FEATURE_COLUMNS

# Optional fallback priors for deployments without historical squad/ranking
# data. They default to zero because the shipped historical snapshots let the
# model learn these effects in backtests instead of applying manual nudges.
DEFAULT_PRIOR_ADJUSTMENTS = {
    "squad_coef": 0.0,
    "squad_clip": 0.25,
    "rank_coef": 0.0,
    "rank_clip": 0.25,
    "chemistry_coef": 0.0,
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
        calibration_temperature: float | None = None,
    ):
        parameters = parameters or {}
        defaults = {
            "loss": "poisson",
            "learning_rate": 0.05,
            "max_iter": 250,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
            "early_stopping": True,
            "random_state": 42,
        }
        defaults.update(parameters)
        self.parameters = defaults
        self.adjustments = {**DEFAULT_PRIOR_ADJUSTMENTS, **(adjustments or {})}
        self.fixed_temperature = calibration_temperature
        self.rho = 0.0
        self.dispersion = 0.0
        self.temperature = calibration_temperature or 1.0
        self.elo_baseline: LogisticRegression | None = None
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.learned_features: set[str] = set()
        self.goal_model = self._new_model()

    def _new_model(self):
        return HistGradientBoostingRegressor(**self.parameters)

    @staticmethod
    def score_matrix(
        home_rate: float,
        away_rate: float,
        max_goals: int = 10,
        rho: float = 0.0,
        dispersion: float = 0.0,
    ):
        goals = np.arange(max_goals + 1)
        if dispersion <= 1e-8:
            home_pmf = poisson.pmf(goals, home_rate)
            away_pmf = poisson.pmf(goals, away_rate)
        else:
            shape = 1.0 / dispersion
            home_pmf = nbinom.pmf(goals, shape, shape / (shape + home_rate))
            away_pmf = nbinom.pmf(goals, shape, shape / (shape + away_rate))
        matrix = np.outer(home_pmf, away_pmf)
        matrix[0, 0] *= 1 - home_rate * away_rate * rho
        matrix[0, 1] *= 1 + home_rate * rho
        matrix[1, 0] *= 1 + away_rate * rho
        matrix[1, 1] *= 1 - rho
        matrix = np.clip(matrix, 0, None)
        return matrix / matrix.sum()

    @staticmethod
    def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
        logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
        logits -= logits.max()
        scaled = np.exp(logits)
        return scaled / scaled.sum()

    def _calibrate_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if np.isclose(self.temperature, 1.0):
            return matrix
        masks = (
            np.tril(np.ones_like(matrix, dtype=bool), -1),
            np.eye(matrix.shape[0], dtype=bool),
            np.triu(np.ones_like(matrix, dtype=bool), 1),
        )
        raw = np.asarray([matrix[mask].sum() for mask in masks])
        calibrated = self._temperature_scale(raw, self.temperature)
        adjusted = matrix.copy()
        for mask, before, after in zip(masks, raw, calibrated, strict=False):
            if before > 0:
                adjusted[mask] *= after / before
        return adjusted / adjusted.sum()

    def outcome_probabilities(self, home_rate: float, away_rate: float):
        matrix = self.score_matrix(
            home_rate,
            away_rate,
            rho=self.rho,
            dispersion=self.dispersion,
        )
        matrix = self._calibrate_matrix(matrix)
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
        flipped = flip_features(features)
        # One model sees every training match from both team perspectives.
        # Predicting the original and flipped rows therefore enforces exact
        # home/away-label symmetry without averaging two independently fit models.
        home_design = features[self.feature_columns].copy()
        away_design = flipped[self.feature_columns].copy()
        learned = getattr(self, "learned_features", set(self.feature_columns))
        inactive = set(self.feature_columns) - set(learned)
        if inactive:
            home_design[list(inactive)] = 0.0
            away_design[list(inactive)] = 0.0
        home = self.goal_model.predict(home_design)
        away = self.goal_model.predict(away_design)
        adjustments = getattr(self, "adjustments", DEFAULT_PRIOR_ADJUSTMENTS)
        if "home_squad_attack" not in learned:
            coef, clip = adjustments["squad_coef"], adjustments["squad_clip"]
            home_edge = (
                features["home_squad_attack"] - features["away_squad_defense"]
            ).fillna(0)
            away_edge = (
                features["away_squad_attack"] - features["home_squad_defense"]
            ).fillna(0)
            home *= np.exp(np.clip(coef * home_edge, -clip, clip))
            away *= np.exp(np.clip(coef * away_edge, -clip, clip))
        if "home_rank_points" not in learned:
            coef, clip = adjustments["rank_coef"], adjustments["rank_clip"]
            rank_edge = features["rank_points_diff"].fillna(0)
            rank_adjustment = np.clip(coef * rank_edge, -clip, clip)
            home *= np.exp(rank_adjustment)
            away *= np.exp(-rank_adjustment)
        if "home_chemistry" not in learned:
            coef, clip = adjustments["chemistry_coef"], adjustments["chemistry_clip"]
            chemistry_edge = features["chemistry_diff"].fillna(0).clip(-clip, clip)
            home *= np.exp(coef * chemistry_edge)
            away *= np.exp(-coef * chemistry_edge)
        home = np.clip(home, 0.05, 6.0)
        away = np.clip(away, 0.05, 6.0)
        return home, away

    def _fit_goal_model(
        self,
        frame: pd.DataFrame,
        weights: pd.Series,
    ) -> None:
        self.feature_columns = FEATURE_COLUMNS.copy()
        minimum_observations = max(100, int(np.ceil(0.01 * len(frame))))
        self.learned_features = {
            column
            for column in FEATURE_COLUMNS
            if frame[column].notna().sum() >= minimum_observations
        }
        original = frame[self.feature_columns].copy()
        inactive = set(self.feature_columns) - self.learned_features
        if inactive:
            original[list(inactive)] = 0.0
        flipped = flip_features(original)
        design = pd.concat(
            [original, flipped],
            ignore_index=True,
        )
        target = pd.concat(
            [frame["home_score"], frame["away_score"]],
            ignore_index=True,
        )
        stacked_weights = np.concatenate([weights.to_numpy(), weights.to_numpy()])
        self.goal_model = self._new_model()
        self.goal_model.fit(design, target, sample_weight=stacked_weights)

    def _fit_dispersion(
        self,
        frame: pd.DataFrame,
        home_rates: np.ndarray,
        away_rates: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        home_goals = frame["home_score"].to_numpy(int)
        away_goals = frame["away_score"].to_numpy(int)

        def objective(dispersion):
            shape = 1.0 / dispersion
            home_probability = shape / (shape + home_rates)
            away_probability = shape / (shape + away_rates)
            log_probability = nbinom.logpmf(
                home_goals,
                shape,
                home_probability,
            ) + nbinom.logpmf(
                away_goals,
                shape,
                away_probability,
            )
            return float(-np.sum(weights * log_probability))

        result = minimize_scalar(
            objective,
            bounds=(1e-4, 1.0),
            method="bounded",
        )
        return float(result.x)

    def _fit_temperature(self, frame: pd.DataFrame) -> float:
        home_rates, away_rates = self._rates(frame)
        outcomes = self._outcome_classes(frame)
        raw = []
        previous = self.temperature
        self.temperature = self.fixed_temperature or 1.0
        for home_rate, away_rate in zip(home_rates, away_rates, strict=False):
            home, draw, away, _ = self.outcome_probabilities(home_rate, away_rate)
            raw.append((home, draw, away))
        self.temperature = previous
        raw = np.asarray(raw)

        def objective(temperature):
            probabilities = np.asarray(
                [
                    self._temperature_scale(row, temperature)
                    for row in raw
                ]
            )
            return float(log_loss(outcomes, probabilities, labels=[0, 1, 2]))

        return float(
            minimize_scalar(
                objective,
                bounds=(0.5, 3.0),
                method="bounded",
            ).x
        )

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
        weights = self._sample_weights(frame, half_life_years)
        self._fit_goal_model(frame, weights)
        home_rates, away_rates = self._rates(frame)
        self.dispersion = self._fit_dispersion(
            frame,
            home_rates,
            away_rates,
            weights.to_numpy(),
        )
        self.rho = self._fit_rho(frame, home_rates, away_rates, weights.to_numpy())
        # Honor a pre-fitted temperature when one is supplied (e.g. the value in
        # config.toml calibrated on a held-out tournament). fit_window has no
        # internal validation split of its own to fit temperature against, so it
        # would otherwise leave the model uncalibrated (temperature 1.0).
        self.temperature = (
            self.fixed_temperature if self.fixed_temperature is not None else 1.0
        )
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
        weights = self._sample_weights(train, half_life_years)
        self._fit_goal_model(train, weights)
        train_home_rates, train_away_rates = self._rates(train)
        self.dispersion = self._fit_dispersion(
            train,
            train_home_rates,
            train_away_rates,
            weights.to_numpy(),
        )
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
        self.temperature = (
            self.fixed_temperature
            if self.fixed_temperature is not None
            else self._fit_temperature(valid)
        )
        metrics = self.evaluate(valid)

        full_weights = self._sample_weights(frame, half_life_years)
        calibrated_temperature = self.temperature
        self._fit_goal_model(frame, full_weights)
        full_home_rates, full_away_rates = self._rates(frame)
        self.dispersion = self._fit_dispersion(
            frame,
            full_home_rates,
            full_away_rates,
            full_weights.to_numpy(),
        )
        self.rho = self._fit_rho(
            frame, full_home_rates, full_away_rates, full_weights.to_numpy()
        )
        self.temperature = calibrated_temperature
        metrics["training_matches"] = int(len(frame))
        metrics["validation_matches"] = int(len(valid))
        metrics["training_through"] = frame["date"].max().date().isoformat()
        metrics["dixon_coles_rho"] = self.rho
        metrics["goal_dispersion"] = self.dispersion
        metrics["calibration_temperature"] = self.temperature
        metrics["active_features"] = sorted(self.learned_features)
        metrics["model_features"] = self.feature_columns
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
