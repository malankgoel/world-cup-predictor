from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import pandas as pd

from . import logbook, report
from .data import (
    RESULT_COLUMNS,
    download_results,
    download_schedule,
    download_shootouts,
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
    feature_path = Path(paths.get("match_features", ""))
    if feature_path.is_file() and feature_path.stat().st_size:
        external = pd.read_csv(feature_path)
        keys = ["date", "home_team", "away_team"]
        missing = set(keys) - set(external.columns)
        if missing:
            raise ValueError(
                f"{feature_path} is missing match keys: {sorted(missing)}"
            )
        external["date"] = pd.to_datetime(external["date"])
        external["home_team"] = external["home_team"].map(normalize_team)
        external["away_team"] = external["away_team"].map(normalize_team)
        value_columns = [
            column
            for column in (
                "home_xg",
                "away_xg",
                "home_odds",
                "draw_odds",
                "away_odds",
            )
            if column in external
        ]
        results = results.merge(
            external[keys + value_columns],
            on=keys,
            how="left",
            suffixes=("", "_external"),
        )
        for column in value_columns:
            external_column = f"{column}_external"
            if external_column in results:
                if column in results:
                    results[column] = results[column].fillna(
                        results[external_column]
                    )
                else:
                    results[column] = results[external_column]
                results = results.drop(columns=external_column)
    rankings = load_optional(paths["rankings"], RANKING_COLUMNS)
    squads = load_optional(paths["squads"], SQUAD_COLUMNS)
    builder = FeatureBuilder(
        rankings,
        squads,
        form_matches=config["training"]["form_matches"],
        state_parameters=config.get("state"),
        elo_parameters=config.get("elo"),
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
    model = WorldCupModel(
        parameters,
        adjustments=config.get("priors"),
        calibration_temperature=config.get("calibration", {}).get(
            "temperature"
        ),
    )
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
        builder.apply_fixture_context(match, states)
    output = pd.DataFrame(rows).round(3)
    path = Path(config["paths"]["predictions"])
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    logbook.record(config, results, "predict", output)
    report.write_report(config, logbook.latest_result_date(results))
    return output


def record_match_xg(config, args) -> bool:
    """Upsert this match's xG into the match-features file. No-op without xG."""
    if args.home_xg is None or args.away_xg is None:
        return False
    mf_path = Path(config["paths"].get("match_features", ""))
    if not str(mf_path):
        return False
    columns = ["date", "home_team", "away_team", "home_xg", "away_xg"]
    if mf_path.is_file() and mf_path.stat().st_size:
        frame = pd.read_csv(mf_path)
    else:
        frame = pd.DataFrame(columns=columns)
    home = normalize_team(args.home)
    away = normalize_team(args.away)
    key = (
        (frame["date"].astype(str) == str(args.date))
        & (frame["home_team"].map(normalize_team) == home)
        & (frame["away_team"].map(normalize_team) == away)
    )
    if key.any():
        frame.loc[key, "home_xg"] = args.home_xg
        frame.loc[key, "away_xg"] = args.away_xg
    else:
        new_row = {
            "date": args.date,
            "home_team": home,
            "away_team": away,
            "home_xg": args.home_xg,
            "away_xg": args.away_xg,
        }
        new_frame = pd.DataFrame([new_row])
        frame = (
            new_frame
            if frame.empty
            else pd.concat([frame, new_frame], ignore_index=True)
        )
    mf_path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values("date").to_csv(mf_path, index=False)
    return True


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
    if "winner" not in frame.columns:
        frame["winner"] = ""
    # An all-empty winner column is read back as float NaN; coerce it to text so
    # writing a (possibly empty) winner string can't trip a dtype error on
    # newer pandas (LossySetitemError / "Invalid value '' for dtype float64").
    frame["winner"] = frame["winner"].fillna("").astype(object)
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
    xg_recorded = record_match_xg(config, args)
    if args.retrain:
        metrics = train(config)
        metrics["xg_recorded"] = xg_recorded
        return metrics
    results, builder = inputs(config)
    save_team_state(config, results, builder)
    return {
        "result_saved": True,
        "model_retrained": False,
        "team_state_updated": True,
        "xg_recorded": xg_recorded,
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
        team_strength_scale=config["simulation"].get(
            "team_strength_scale",
            0.20,
        ),
        penalty_skill_weight=config["simulation"].get(
            "penalty_skill_weight",
            0.8927,
        ),
    )
    output = simulator.simulate(
        runs or config["simulation"]["runs"],
        seed if seed is not None else config["simulation"]["seed"],
    ).round(3)
    path = Path(config["paths"]["simulation"])
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    logbook.record(config, results, "simulate", output)
    report.write_report(config, logbook.latest_result_date(results))
    return output


def show_log(config) -> list[dict]:
    """Print the recorded forecast history, oldest first."""
    entries = logbook.history(config)
    if not entries:
        print("No forecasts logged yet. Run `worldcup predict` or `simulate`.")
        return entries
    for entry in entries:
        when = entry.get("last_run_at") or entry.get("recorded_at")
        line = (
            f"{when}  {entry['kind']:8s} "
            f"through={entry['results_through']}  "
            f"played={entry['played_matches']}"
        )
        if entry.get("top_champions"):
            leader = entry["top_champions"][0]
            line += f"  top={leader['team']} ({leader['champion']:.3f})"
        print(line)
    return entries


def backtest(config, cutoffs: str | None, out: str | None):
    from .backtest import expanding_window_backtest, summarize_backtest

    results, builder = inputs(config)
    cutoff = cutoff_date(config)
    if cutoff is not None:
        results = results[results["date"] < cutoff].reset_index(drop=True)
    features = builder.training_frame(results)
    if cutoffs:
        fold_cutoffs = [value.strip() for value in cutoffs.split(",")]
    else:
        last_year = int(features["date"].max().year)
        fold_cutoffs = [f"{year}-01-01" for year in range(last_year - 3, last_year + 1)]
    parameters = dict(config["model"])
    parameters["random_state"] = config["training"]["random_state"]
    table = expanding_window_backtest(
        features,
        cutoffs=fold_cutoffs,
        model_parameters=parameters,
        model_adjustments=config.get("priors"),
        calibration_temperature=config.get("calibration", {}).get(
            "temperature"
        ),
        half_life_years=config["training"]["recency_half_life_years"],
    )
    path = Path(out or "outputs/backtest.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(path, index=False)
    return table, summarize_backtest(table)


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
    backtest_command = commands.add_parser("backtest")
    backtest_command.add_argument(
        "--cutoffs",
        help="Comma-separated fold-boundary dates, e.g. 2023-01-01,2024-01-01. "
        "Defaults to January 1 of each of the last four seasons.",
    )
    backtest_command.add_argument("--out", help="CSV output path for per-fold metrics")
    commands.add_parser("log")
    commands.add_parser("report")
    update = commands.add_parser("update")
    update.add_argument("--date", required=True)
    update.add_argument("--home", required=True)
    update.add_argument("--away", required=True)
    update.add_argument("--home-score", required=True, type=int)
    update.add_argument("--away-score", required=True, type=int)
    update.add_argument(
        "--home-xg",
        type=float,
        help="Home expected goals for this match; recorded so the team-state "
        "update reflects performance, not just the scoreline",
    )
    update.add_argument(
        "--away-xg",
        type=float,
        help="Away expected goals for this match (see --home-xg)",
    )
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
        if config["paths"].get("shootouts"):
            print(download_shootouts(config["paths"]["shootouts"]))
    elif args.command == "train":
        print(json.dumps(train(config), indent=2))
    elif args.command == "predict":
        print(predict_schedule(config).to_string(index=False))
    elif args.command == "update":
        print(json.dumps(update_result(config, args), indent=2))
    elif args.command == "simulate":
        print(simulate(config, args.runs, args.seed).head(20).to_string(index=False))
    elif args.command == "backtest":
        table, summary = backtest(config, args.cutoffs, args.out)
        print(table.round(4).to_string(index=False))
        print("\nmean across folds:")
        print(json.dumps(summary, indent=2))
    elif args.command == "log":
        show_log(config)
    elif args.command == "report":
        results, _ = inputs(config)
        path = report.write_report(config, logbook.latest_result_date(results))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
