"""Calibrate tournament-wide latent strength spread on the 2022 World Cup."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldcup_predictor.cli import inputs, load_config  # noqa: E402
from worldcup_predictor.data import load_schedule  # noqa: E402
from worldcup_predictor.model import WorldCupModel  # noqa: E402
from worldcup_predictor.tournament import TournamentSimulator  # noqa: E402

KICKOFF = pd.Timestamp("2022-11-20")
ACTUAL = {
    "reach_round_of_16": {
        "Netherlands",
        "Senegal",
        "England",
        "United States",
        "Argentina",
        "Poland",
        "France",
        "Australia",
        "Japan",
        "Spain",
        "Morocco",
        "Croatia",
        "Brazil",
        "Switzerland",
        "Portugal",
        "South Korea",
    },
    "reach_quarter_final": {
        "Netherlands",
        "Argentina",
        "France",
        "England",
        "Croatia",
        "Brazil",
        "Morocco",
        "Portugal",
    },
    "reach_semi_final": {"Argentina", "France", "Croatia", "Morocco"},
    "reach_final": {"Argentina", "France"},
    "champion": {"Argentina"},
}


def main(runs: str = "750") -> None:
    config = load_config(ROOT / "config.toml")
    results, builder = inputs(config)
    results = results[results["date"] < KICKOFF].reset_index(drop=True)
    frame = builder.training_frame(results)
    parameters = {
        **config["model"],
        "random_state": config["training"]["random_state"],
    }
    model = WorldCupModel(
        parameters,
        adjustments=config.get("priors"),
        calibration_temperature=config.get("calibration", {}).get(
            "temperature"
        ),
    )
    model.fit_window(
        frame,
        half_life_years=config["training"]["recency_half_life_years"],
    )
    schedule = load_schedule(ROOT / "data" / "input" / "schedule_2022.csv")
    simulator = TournamentSimulator(
        model,
        builder,
        results,
        schedule,
        penalty_skill_weight=config["simulation"]["penalty_skill_weight"],
    )
    records = []
    for scale in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        simulator.team_strength_scale = scale
        table = simulator.simulate(
            runs=int(runs),
            seed=config["simulation"]["seed"],
        ).set_index("team")
        stage_brier = {}
        for stage, actual in ACTUAL.items():
            probability = table[stage].to_numpy()
            observed = table.index.isin(actual).astype(float)
            stage_brier[stage] = float(np.mean((probability - observed) ** 2))
        records.append(
            {
                "team_strength_scale": scale,
                "mean_stage_brier": float(np.mean(list(stage_brier.values()))),
                "stage_brier": stage_brier,
            }
        )
        print(
            f"scale={scale:.2f} mean_stage_brier="
            f"{records[-1]['mean_stage_brier']:.6f}",
            flush=True,
        )
    records.sort(key=lambda row: row["mean_stage_brier"])
    payload = {"runs": int(runs), "best": records[0], "candidates": records}
    output = ROOT / "outputs" / "team_strength_calibration.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(records[0], indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
