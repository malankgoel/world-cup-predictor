"""Fit the tournament shootout-strength weight from historical shootouts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldcup_predictor.cli import cutoff_date, inputs, load_config  # noqa: E402
from worldcup_predictor.data import normalize_team  # noqa: E402


def main() -> None:
    config = load_config(ROOT / "config.toml")
    results, builder = inputs(config)
    cutoff = cutoff_date(config)
    if cutoff is not None:
        results = results[results["date"] < cutoff]
    shootouts = pd.read_csv(config["paths"]["shootouts"])
    shootouts["date"] = pd.to_datetime(shootouts["date"])
    for column in ("home_team", "away_team", "winner"):
        shootouts[column] = shootouts[column].map(normalize_team)
    winners = {
        (row.date, row.home_team, row.away_team): row.winner
        for row in shootouts.itertuples()
    }

    states = {}
    elo_differences = []
    home_won = []
    for _, match in results.sort_values("date").iterrows():
        key = (
            pd.Timestamp(match["date"]),
            normalize_team(match["home_team"]),
            normalize_team(match["away_team"]),
        )
        winner = winners.get(key)
        if winner:
            features = builder.make_features(match, states)
            elo_differences.append(features["elo_diff"])
            home_won.append(float(winner == key[1]))
        builder.apply_result(match, states)

    elo_differences = np.asarray(elo_differences)
    home_won = np.asarray(home_won)
    elo_probability = 1.0 / (1.0 + np.exp(-elo_differences / 400.0))

    def objective(weight):
        probability = 0.5 + weight * (elo_probability - 0.5)
        probability = np.clip(probability, 1e-6, 1 - 1e-6)
        return float(
            -np.mean(
                home_won * np.log(probability)
                + (1 - home_won) * np.log(1 - probability)
            )
        )

    result = minimize_scalar(objective, bounds=(0.0, 1.5), method="bounded")
    payload = {
        "shootouts": int(len(home_won)),
        "penalty_skill_weight": float(result.x),
        "log_loss": float(result.fun),
        "coin_flip_log_loss": float(np.log(2)),
    }
    output = ROOT / "outputs" / "shootout_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
