"""Random search over model, latent-state, and Elo constants using backtests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldcup_predictor.backtest import (  # noqa: E402
    expanding_window_backtest,
    summarize_backtest,
)
from worldcup_predictor.cli import cutoff_date, inputs, load_config  # noqa: E402
from worldcup_predictor.features import FeatureBuilder  # noqa: E402

CUTOFFS = ["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"]


def choice(rng, values):
    return values[int(rng.integers(0, len(values)))]


def candidate(config, rng, baseline: bool) -> dict:
    if baseline:
        return {
            "model": dict(config["model"]),
            "state": dict(config["state"]),
            "elo": dict(config["elo"]),
        }
    model = dict(config["model"])
    model.update(
        learning_rate=choice(rng, [0.03, 0.05, 0.08]),
        max_iter=choice(rng, [150, 250, 350]),
        max_leaf_nodes=choice(rng, [7, 15, 31]),
        l2_regularization=choice(rng, [0.5, 1.0, 3.0]),
        early_stopping=True,
    )
    state = dict(config["state"])
    state.update(
        neutral_base_log=choice(
            rng,
            [
                config["state"]["neutral_base_log"] - 0.05,
                config["state"]["neutral_base_log"],
                config["state"]["neutral_base_log"] + 0.05,
            ],
        ),
        home_advantage_log=choice(rng, [0.16, 0.214, 0.27]),
        observation_variance=choice(rng, [1.0, 1.5, 2.0]),
        update_step=choice(rng, [0.35, 0.5, 0.65]),
        surprise_clip=choice(rng, [1.0, 1.2, 1.5]),
    )
    elo = dict(config["elo"])
    elo.update(
        k_factor=choice(rng, [20.0, 28.0, 36.0]),
        home_advantage_points=choice(rng, [60.0, 80.0, 100.0]),
    )
    return {"model": model, "state": state, "elo": elo}


def main(trials: str = "8") -> None:
    config = load_config(ROOT / "config.toml")
    results, _ = inputs(config)
    cutoff = cutoff_date(config)
    if cutoff is not None:
        results = results[results["date"] < cutoff].reset_index(drop=True)
    # Reuse parsed lookup source data by loading the configured CSVs once per
    # trial; feature generation is what changes with state/Elo constants.
    import pandas as pd

    ranking_frame = pd.read_csv(config["paths"]["rankings"])
    squad_frame = pd.read_csv(config["paths"]["squads"])
    rng = np.random.default_rng(config["training"]["random_state"])
    records = []
    for index in range(int(trials)):
        settings = candidate(config, rng, baseline=index == 0)
        builder = FeatureBuilder(
            ranking_frame,
            squad_frame,
            form_matches=config["training"]["form_matches"],
            state_parameters=settings["state"],
            elo_parameters=settings["elo"],
        )
        frame = builder.training_frame(results)
        table = expanding_window_backtest(
            frame,
            cutoffs=CUTOFFS,
            model_parameters={
                **settings["model"],
                "random_state": config["training"]["random_state"],
            },
            model_adjustments=config.get("priors"),
            calibration_temperature=config.get("calibration", {}).get(
                "temperature"
            ),
            half_life_years=config["training"]["recency_half_life_years"],
        )
        summary = summarize_backtest(table)
        objective = (
            summary["ranked_probability_score"]
            + 0.2 * summary["outcome_log_loss"]
        )
        record = {
            "trial": index,
            "objective": objective,
            "summary": summary,
            **settings,
        }
        records.append(record)
        print(
            f"trial {index}: objective={objective:.6f} "
            f"rps={summary['ranked_probability_score']:.6f} "
            f"logloss={summary['outcome_log_loss']:.6f}",
            flush=True,
        )
    records.sort(key=lambda row: row["objective"])
    payload = {"cutoffs": CUTOFFS, "best": records[0], "trials": records}
    output = ROOT / "outputs" / "parameter_search.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(records[0], indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
