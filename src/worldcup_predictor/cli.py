from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import pandas as pd

from .data import (
    RESULT_COLUMNS,
    download_results,
    download_schedule,
    load_optional,
    load_results,
    load_schedule,
    normalize_team,
)
from .features import FeatureBuilder
from .model import WorldCupModel
from .tournament import TournamentSimulator

RANKING_COLUMNS = ["date", "team", "rank", "points"]
SQUAD_COLUMNS = [
    "as_of",
    "team",
    "player",
    "club",
    "position",
    "talent",
    "talent_se",
    "exp_minutes",
    "available",
    "age",
    "is_starter",
]


def load_config(path: str) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def cutoff_date(config) -> pd.Timestamp | None:
    """Exclusive upper bound on result dates, or None to include everything.

    Returns None when include_tournament is set (phase 2), so tournament
    games flow into training and team state; otherwise the configured
    cutoff keeps the model purely pre-tournament.
    """
    training = config["training"]
    if training.get("include_tournament", False):
        return None
    value = training.get("cutoff_date")
    return pd.Timestamp(value) if value else None


def inputs(config):
    paths = config["paths"]
    results = load_results(
        paths["results"], start_date=config["training"]["start_date"]
    )
    rankings = load_optional(paths["rankings"], RANKING_COLUMNS)
    squads = load_optional(paths["squads"], SQUAD_COLUMNS)
    builder = FeatureBuilder(
        rankings, squads, form_matches=config["training"]["form_matches"]
    )
    return results, builder


def save_team_state(config, results, builder) -> None:
    states = builder.build_states(results, before=cutoff_date(config))
    payload = {
        team: {
            "elo": state.elo,
            "attack_mean": state.attack_mean,
            "attack_var": state.attack_var,
            "defense_mean": state.defense_mean,
            "defense_var": state.defense_var,
            "matches": state.played,
            "last_updated": (
                state.last_date.date().isoformat() if state.last_date is not None else None
            ),
        }
        for team, state in sorted(states.items())
    }
    path = Path(config["paths"]["team_state"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def train(config) -> dict:
    results, builder = inputs(config)
    cutoff = cutoff_date(config)
    if cutoff is not None:
        results = results[results["date"] < cutoff].reset_index(drop=True)
    frame = builder.training_frame(results)
    parameters = dict(config["model"])
    parameters["random_state"] = config["training"]["random_state"]
    model = WorldCupModel(parameters)
    metrics = model.fit(
        frame,
        validation_fraction=config["training"]["validation_fraction"],
        half_life_years=config["training"]["recency_half_life_years"],
    )
    model.save(config["paths"]["model"])
    save_team_state(config, results, builder)
    metrics_path = Path(config["paths"]["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def predict_schedule(config) -> pd.DataFrame:
    results, builder = inputs(config)
    schedule = load_schedule(config["paths"]["schedule"])
    model = WorldCupModel.load(config["paths"]["model"])
    completed = {
        (
            pd.Timestamp(row.date).date(),
            normalize_team(row.home_team),
            normalize_team(row.away_team),
        )
        for row in results.itertuples()
    }
    states = builder.build_states(results, before=cutoff_date(config))
    rows = []
    for _, fixture in schedule[schedule["stage"] == "group"].iterrows():
        key = (
            fixture["date"].date(),
            fixture["home_team"],
            fixture["away_team"],
        )
        if key in completed:
            continue
        match = fixture.copy()
        match["tournament"] = "FIFA World Cup"
        features = builder.make_features(match, states)
        prediction = model.predict(features)
        rows.append(
            {
                "match_id": fixture["match_id"],
                "date": fixture["date"].date().isoformat(),
                "group": fixture["group"],
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                **prediction.__dict__,
            }
        )
    output = pd.DataFrame(rows).round(3)
    path = Path(config["paths"]["predictions"])
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


def update_result(config, args) -> dict:
    path = Path(config["paths"]["results"])
    frame = pd.read_csv(path)
    row = {
        "date": args.date,
        "home_team": normalize_team(args.home),
        "away_team": normalize_team(args.away),
        "home_score": args.home_score,
        "away_score": args.away_score,
        "tournament": args.tournament,
        "city": args.city,
        "country": args.country,
        "neutral": args.neutral,
        "winner": normalize_team(args.winner) if args.winner else "",
    }
    if "winner" not in frame:
        frame["winner"] = ""
    key = (
        (frame["date"] == row["date"])
        & (frame["home_team"].map(normalize_team) == row["home_team"])
        & (frame["away_team"].map(normalize_team) == row["away_team"])
    )
    if key.any():
        for column in RESULT_COLUMNS:
            frame.loc[key, column] = row[column]
        frame.loc[key, "winner"] = row["winner"]
    else:
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame.sort_values("date").to_csv(path, index=False)
    if args.retrain:
        return train(config)
    results, builder = inputs(config)
    save_team_state(config, results, builder)
    return {
        "result_saved": True,
        "model_retrained": False,
        "team_state_updated": True,
        "latest_result_date": args.date,
    }


def simulate(config, runs: int | None, seed: int | None) -> pd.DataFrame:
    results, builder = inputs(config)
    schedule = load_schedule(config["paths"]["schedule"])
    model = WorldCupModel.load(config["paths"]["model"])
    simulator = TournamentSimulator(
        model,
        builder,
        results,
        schedule,
        include_tournament=config["training"].get("include_tournament", False),
    )
    output = simulator.simulate(
        runs or config["simulation"]["runs"],
        seed if seed is not None else config["simulation"]["seed"],
    ).round(3)
    path = Path(config["paths"]["simulation"])
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="worldcup")
    root.add_argument("--config", default="config.toml")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("download-data")
    commands.add_parser("train")
    commands.add_parser("predict")
    simulation = commands.add_parser("simulate")
    simulation.add_argument("--runs", type=int)
    simulation.add_argument("--seed", type=int)
    update = commands.add_parser("update")
    update.add_argument("--date", required=True)
    update.add_argument("--home", required=True)
    update.add_argument("--away", required=True)
    update.add_argument("--home-score", required=True, type=int)
    update.add_argument("--away-score", required=True, type=int)
    update.add_argument("--tournament", default="FIFA World Cup")
    update.add_argument("--city", default="")
    update.add_argument("--country", default="")
    update.add_argument(
        "--neutral", action=argparse.BooleanOptionalAction, default=True
    )
    update.add_argument(
        "--retrain",
        action="store_true",
        help="Refit the global goal model; live Bayesian state updates do not require it",
    )
    update.add_argument(
        "--winner",
        help="Advancing team when a completed knockout match was tied after extra time",
    )
    return root


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    if args.command == "download-data":
        print(download_results(config["paths"]["results"]))
        print(download_schedule(config["paths"]["schedule"]))
    elif args.command == "train":
        print(json.dumps(train(config), indent=2))
    elif args.command == "predict":
        print(predict_schedule(config).to_string(index=False))
    elif args.command == "update":
        print(json.dumps(update_result(config, args), indent=2))
    elif args.command == "simulate":
        print(simulate(config, args.runs, args.seed).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
