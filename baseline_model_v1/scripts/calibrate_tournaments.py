"""Fit 1X2 temperature scaling on the held-out 2018 World Cup.

Temperature is fit on the 2018 World Cup ONLY and then validated on the 2022
World Cup, which the fit never sees. This keeps 2022 a clean out-of-sample check
of the calibration (the previous version fit on 2018 + 2022 jointly, so the 2022
backtest could not honestly measure calibration). Writes the fitted temperature
and the raw-vs-calibrated log loss on both the fit and validation tournaments.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldcup_predictor.cli import inputs, load_config  # noqa: E402
from worldcup_predictor.model import WorldCupModel  # noqa: E402

# The tournament the temperature is fit on, and the later one held out to
# validate it. 2022 is never used during fitting.
FIT_TOURNAMENT = (pd.Timestamp("2018-06-14"), pd.Timestamp("2018-07-16"))
VALIDATION_TOURNAMENT = (pd.Timestamp("2022-11-20"), pd.Timestamp("2022-12-19"))


def _tournament_probabilities(
    frame: pd.DataFrame,
    config: dict,
    kickoff: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """Train on everything before `kickoff`, return raw 1X2 probs + outcomes."""
    train = frame[frame["date"] < kickoff]
    test = frame[
        (frame["date"] >= kickoff)
        & (frame["date"] < end)
        & (frame["tournament"] == "FIFA World Cup")
    ]
    parameters = {
        **config["model"],
        "random_state": config["training"]["random_state"],
    }
    model = WorldCupModel(parameters, adjustments=config.get("priors"))
    model.fit_window(
        train,
        half_life_years=config["training"]["recency_half_life_years"],
    )
    # Force raw (uncalibrated) probabilities so temperature is fit cleanly.
    model.temperature = 1.0
    home_rates, away_rates = model._rates(test)
    probabilities = np.asarray(
        [
            model.outcome_probabilities(home_rate, away_rate)[:3]
            for home_rate, away_rate in zip(home_rates, away_rates, strict=False)
        ]
    )
    return probabilities, model._outcome_classes(test)


def _scaled(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    return np.asarray(
        [
            WorldCupModel._temperature_scale(row, temperature)
            for row in probabilities
        ]
    )


def main() -> None:
    config = load_config(ROOT / "config.toml")
    results, builder = inputs(config)
    results = results[results["date"] < VALIDATION_TOURNAMENT[1]].reset_index(
        drop=True
    )
    ordered = results.sort_values("date").reset_index(drop=True)
    frame = builder.training_frame(ordered)
    frame["tournament"] = ordered["tournament"].to_numpy()

    fit_probabilities, fit_outcomes = _tournament_probabilities(
        frame, config, *FIT_TOURNAMENT
    )
    validation_probabilities, validation_outcomes = _tournament_probabilities(
        frame, config, *VALIDATION_TOURNAMENT
    )

    result = minimize_scalar(
        lambda temperature: float(
            log_loss(
                fit_outcomes,
                _scaled(fit_probabilities, temperature),
                labels=[0, 1, 2],
            )
        ),
        bounds=(0.5, 3.0),
        method="bounded",
    )
    temperature = float(result.x)

    def log_losses(probabilities, outcomes):
        return {
            "raw": float(log_loss(outcomes, probabilities, labels=[0, 1, 2])),
            "calibrated": float(
                log_loss(
                    outcomes,
                    _scaled(probabilities, temperature),
                    labels=[0, 1, 2],
                )
            ),
        }

    payload = {
        "temperature": temperature,
        "fit_tournament": FIT_TOURNAMENT[0].year,
        "validation_tournament": VALIDATION_TOURNAMENT[0].year,
        "fit_matches": int(len(fit_outcomes)),
        "validation_matches": int(len(validation_outcomes)),
        "fit_log_loss": log_losses(fit_probabilities, fit_outcomes),
        "validation_log_loss": log_losses(
            validation_probabilities, validation_outcomes
        ),
    }
    output = ROOT / "outputs" / "tournament_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
