"""Out-of-sample test of the model on the 2022 World Cup.

Trains only on internationals played BEFORE the 2022 kickoff (no leakage),
then (1) scores its 1X2 / totals predictions against the actual 64 tournament
matches, and (2) simulates the full bracket and compares advancement / title
probabilities to what actually happened.

Run:  python scripts/backtest_2022.py
Writes outputs/backtest_2022_probabilities.csv and outputs/backtest_2022_metrics.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldcup_predictor.cli import inputs, load_config  # noqa: E402
from worldcup_predictor.data import load_schedule  # noqa: E402
from worldcup_predictor.model import WorldCupModel  # noqa: E402
from worldcup_predictor.tournament import TournamentSimulator  # noqa: E402

START_DATE = "2010-01-01"
KICKOFF = pd.Timestamp("2022-11-20")
TOURNAMENT_END = pd.Timestamp("2022-12-20")
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
SEED = 42

# What actually happened in 2022 (for scoring the predictions).
ACTUAL = {
    "reach_round_of_16": {
        "Netherlands", "Senegal", "England", "United States", "Argentina",
        "Poland", "France", "Australia", "Japan", "Spain", "Morocco",
        "Croatia", "Brazil", "Switzerland", "Portugal", "South Korea",
    },
    "reach_quarter_final": {
        "Netherlands", "Argentina", "France", "England",
        "Croatia", "Brazil", "Morocco", "Portugal",
    },
    "reach_semi_final": {"Argentina", "France", "Croatia", "Morocco"},
    "reach_final": {"Argentina", "France"},
    "champion": {"Argentina"},
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def binary_log_loss(prob: np.ndarray, actual: np.ndarray) -> float:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return float(-np.mean(actual * np.log(prob) + (1 - actual) * np.log(1 - prob)))


def main() -> None:
    log(f"loading results since {START_DATE}")
    config = load_config(ROOT / "config.toml")
    results, builder = inputs(config)
    results = results[results["date"] >= pd.Timestamp(START_DATE)]
    results = results[results["date"] < TOURNAMENT_END].reset_index(drop=True)
    schedule = load_schedule(ROOT / "data" / "input" / "schedule_2022.csv")

    ordered = results.sort_values("date").reset_index(drop=True)
    log(f"building walk-forward features for {len(ordered)} matches")
    frame = builder.training_frame(ordered)
    frame["tournament"] = ordered["tournament"].to_numpy()

    train = frame[frame["date"] < KICKOFF]
    test = frame[
        (frame["date"] >= KICKOFF) & (frame["tournament"] == "FIFA World Cup")
    ]
    log(f"train matches: {len(train)} | 2022 WC test matches: {len(test)}")

    parameters = {
        **config["model"],
        "random_state": SEED,
    }
    model = WorldCupModel(
        parameters,
        adjustments=config.get("priors"),
        calibration_temperature=config.get("calibration", {}).get(
            "temperature"
        ),
    )
    log("fitting pre-2022 goal models")
    model.fit_window(train, half_life_years=4.0)

    log("scoring the 64 actual 2022 matches (match level)")
    match_metrics = model.evaluate(test)

    log(f"simulating the bracket ({RUNS} runs)")
    results_pre = results[results["date"] < KICKOFF].reset_index(drop=True)
    simulator = TournamentSimulator(
        model,
        builder,
        results_pre,
        schedule,
        team_strength_scale=config["simulation"]["team_strength_scale"],
        penalty_skill_weight=config["simulation"]["penalty_skill_weight"],
    )
    table = simulator.simulate(runs=RUNS, seed=SEED).reset_index(drop=True)

    # Score the tournament-level probabilities against what happened.
    stage_scores = {}
    for stage, actual_set in ACTUAL.items():
        prob = table.set_index("team")[stage]
        actual = prob.index.isin(actual_set).astype(float)
        stage_scores[stage] = {
            "brier": float(np.mean((prob.to_numpy() - actual) ** 2)),
            "log_loss": binary_log_loss(prob.to_numpy(), actual),
            "prob_on_actuals": {
                team: round(float(prob[team]), 3) for team in sorted(actual_set)
            },
        }

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out_dir / "backtest_2022_probabilities.csv", index=False)
    report = {
        "config": {
            "start_date": START_DATE,
            "max_iter": parameters["max_iter"],
            "runs": RUNS,
            "train_matches": int(len(train)),
            "test_matches": int(len(test)),
        },
        "match_level": match_metrics,
        "tournament_level": stage_scores,
    }
    (out_dir / "backtest_2022_metrics.json").write_text(json.dumps(report, indent=2))

    # Human-readable summary.
    print("\n================ 2022 WORLD CUP BACKTEST ================")
    print(f"trained on {len(train)} matches since {START_DATE}; "
          f"tested on {len(test)} actual 2022 matches\n")
    print("--- MATCH LEVEL (lower is better) ---")
    for key in (
        "ranked_probability_score", "outcome_log_loss", "brier_score",
        "calibration_error", "elo_baseline_log_loss", "elo_baseline_rps",
        "over_2_5_brier", "both_teams_to_score_brier",
        "home_goal_mae", "away_goal_mae",
    ):
        print(f"  {key:32s} {match_metrics[key]:.4f}")

    print("\n--- TOURNAMENT LEVEL: top 12 by predicted title odds ---")
    top = table.head(12)
    print(f"  {'team':16s} {'R16':>6} {'QF':>6} {'SF':>6} {'Final':>6} {'Champ':>6}")
    for _, row in top.iterrows():
        print(f"  {row['team']:16s} "
              f"{row['reach_round_of_16']:6.2f} {row['reach_quarter_final']:6.2f} "
              f"{row['reach_semi_final']:6.2f} {row['reach_final']:6.2f} "
              f"{row['champion']:6.2f}")

    print("\n--- PREDICTED PROBABILITY ON ACTUAL OUTCOMES ---")
    print(f"  champion Argentina: {stage_scores['champion']['prob_on_actuals']['Argentina']:.3f}")
    print(f"  finalists: {stage_scores['reach_final']['prob_on_actuals']}")
    print(f"  semifinalists: {stage_scores['reach_semi_final']['prob_on_actuals']}")
    print("\n  Brier score by stage (lower is better):")
    for stage, scores in stage_scores.items():
        print(f"    {stage:22s} {scores['brier']:.4f}")
    log("done")


if __name__ == "__main__":
    main()
