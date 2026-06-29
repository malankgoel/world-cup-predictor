from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
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
from .tournament import (
    TournamentSimulator,
    assign_third_place_teams,
)
from .tournament import _rank_group as rank_group

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


def merge_shootout_winners(results: pd.DataFrame, shootouts_path) -> pd.DataFrame:
    """Fill the ``winner`` column from the shootouts file for penalty-decided
    games.

    The martj42 results feed has no ``winner`` column — penalty-shootout
    outcomes live in a separate shootouts.csv. Without this merge a knockout game
    that was level after extra time has a blank winner, so the simulator falls
    back to an Elo coin-flip instead of pinning who actually advanced. Only blank
    winners are filled, so any manually entered ``--winner`` is preserved.
    """
    if not shootouts_path:
        return results
    path = Path(shootouts_path)
    if not (path.is_file() and path.stat().st_size):
        return results
    shootouts = pd.read_csv(path)
    if not {"date", "home_team", "away_team", "winner"}.issubset(shootouts.columns):
        return results
    shootouts = shootouts[["date", "home_team", "away_team", "winner"]].copy()
    shootouts["date"] = pd.to_datetime(shootouts["date"], errors="coerce")
    shootouts["home_team"] = shootouts["home_team"].map(normalize_team)
    shootouts["away_team"] = shootouts["away_team"].map(normalize_team)
    shootouts["winner"] = shootouts["winner"].map(
        lambda value: normalize_team(value) if isinstance(value, str) and value else ""
    )
    shootouts = shootouts.rename(columns={"winner": "_shootout_winner"})
    results = results.merge(
        shootouts, on=["date", "home_team", "away_team"], how="left"
    )
    blank = results["winner"].fillna("") == ""
    results.loc[blank, "winner"] = results.loc[blank, "_shootout_winner"].fillna("")
    return results.drop(columns="_shootout_winner")


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
    results = merge_shootout_winners(results, paths.get("shootouts"))
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
    parameters["importance_power"] = config["training"].get(
        "importance_weight_power", 0.75
    )
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


def load_odds_map(config) -> dict:
    """Map (date, home, away) -> (home_odds, draw_odds, away_odds) from the
    match-features file, used to anchor published predictions to the market.

    Only rows with all three decimal odds present are returned; team names and
    dates are normalised so lookups match the schedule.
    """
    path = Path(config["paths"].get("match_features", ""))
    if not (str(path) and path.is_file() and path.stat().st_size):
        return {}
    frame = pd.read_csv(path)
    needed = {"date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds"}
    if not needed.issubset(frame.columns):
        return {}
    odds_map: dict = {}
    for row in frame.itertuples():
        triple = (row.home_odds, row.draw_odds, row.away_odds)
        if not all(pd.notna(value) for value in triple):
            continue
        key = (
            pd.Timestamp(row.date).date(),
            normalize_team(row.home_team),
            normalize_team(row.away_team),
        )
        odds_map[key] = tuple(float(value) for value in triple)
    return odds_map


def predict_schedule(config) -> pd.DataFrame:
    results, builder = inputs(config)
    schedule = load_schedule(config["paths"]["schedule"])
    model = WorldCupModel.load(config["paths"]["model"])
    odds_map = load_odds_map(config)
    blend_weight = float(config.get("market", {}).get("blend_weight", 0.0))
    completed = set()
    for row in results.itertuples():
        played_on = pd.Timestamp(row.date).date()
        home = normalize_team(row.home_team)
        away = normalize_team(row.away_team)
        # Both orderings: the feed may list a host/neutral game with sides
        # swapped versus the schedule, and we still want to skip the played game.
        completed.add((played_on, home, away))
        completed.add((played_on, away, home))
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
        odds = odds_map.get(
            (
                fixture["date"].date(),
                normalize_team(fixture["home_team"]),
                normalize_team(fixture["away_team"]),
            )
        )
        if odds is not None:
            match["home_odds"], match["draw_odds"], match["away_odds"] = odds
        features = builder.make_features(match, states)
        prediction = model.predict(features)
        if blend_weight > 0.0:
            blended = WorldCupModel.blend_market(
                (prediction.home_win, prediction.draw, prediction.away_win),
                (
                    features.get("home_market_probability"),
                    features.get("draw_market_probability"),
                    features.get("away_market_probability"),
                ),
                blend_weight,
            )
            prediction.home_win, prediction.draw, prediction.away_win = blended
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


def resolve_first_knockout_matchups(schedule, results, seed: int = 42):
    """Resolve the actual first knockout round (Round of 32 in 2026) from the
    completed group results.

    Ranks every group, assigns the best third-placed teams, and maps each
    bracket slot (``1A``, ``2B``, ``3CDEF`` …) to a real team — the same logic
    the simulator uses, but deterministic on the real scorelines. Raises if any
    group game is still missing from ``results`` (the bracket isn't decided yet).
    Returns ``(fixtures, group_order)`` where fixtures is a list of
    ``(match_id, schedule_row, home_team, away_team)``.
    """
    group_rows = schedule[schedule["stage"] == "group"]
    tournament_start = pd.Timestamp(group_rows["date"].min())
    # Restrict to this tournament so a same-named historical pairing can't shadow
    # a 2026 group game, and key both home/away orderings: the results feed lists
    # a host nation as "home" even when the schedule has it away (e.g. Canada vs
    # Switzerland), so an exact-tuple match would spuriously miss those games.
    world_cup = results[
        (results["tournament"] == "FIFA World Cup")
        & (results["date"] >= tournament_start)
    ]
    scores: dict[tuple[str, str], tuple[int, int]] = {}
    for row in world_cup.itertuples():
        home = normalize_team(row.home_team)
        away = normalize_team(row.away_team)
        hg, ag = int(row.home_score), int(row.away_score)
        scores[(home, away)] = (hg, ag)
        scores[(away, home)] = (ag, hg)
    table: dict[str, dict] = {}
    group_matches: dict[str, list] = {}
    missing = []
    for _, row in group_rows.iterrows():
        group = row["group"]
        home = normalize_team(row["home_team"])
        away = normalize_team(row["away_team"])
        table.setdefault(group, {})
        group_matches.setdefault(group, [])
        for team in (home, away):
            table[group].setdefault(team, {"points": 0, "gf": 0, "ga": 0})
        if (home, away) not in scores:
            missing.append((row["date"].date(), home, away))
            continue
        hg, ag = scores[(home, away)]
        table[group][home]["gf"] += hg
        table[group][home]["ga"] += ag
        table[group][away]["gf"] += ag
        table[group][away]["ga"] += hg
        if hg > ag:
            table[group][home]["points"] += 3
        elif ag > hg:
            table[group][away]["points"] += 3
        else:
            table[group][home]["points"] += 1
            table[group][away]["points"] += 1
        group_matches[group].append((home, away, hg, ag))
    if missing:
        raise ValueError(
            f"{len(missing)} group game(s) not yet in results, so the bracket "
            f"is not decided. First missing: {missing[0][1]} vs {missing[0][2]} "
            f"on {missing[0][0]}."
        )

    rng = np.random.default_rng(seed)
    group_order = {
        group: rank_group(list(teams), teams, group_matches[group], rng)
        for group, teams in table.items()
    }
    knockout_rows = schedule[schedule["stage"] != "group"]
    third_sources = sorted(
        {
            source
            for source in pd.concat(
                [knockout_rows["home_source"], knockout_rows["away_source"]]
            )
            if str(source).startswith("3")
        }
    )
    if third_sources:
        third_order = sorted(
            [(group, teams[2]) for group, teams in group_order.items()],
            key=lambda item: (
                table[item[0]][item[1]]["points"],
                table[item[0]][item[1]]["gf"] - table[item[0]][item[1]]["ga"],
                table[item[0]][item[1]]["gf"],
                rng.random(),
            ),
            reverse=True,
        )
        third_assignments = assign_third_place_teams(
            third_sources, dict(third_order[: len(third_sources)])
        )
    else:
        third_assignments = {}

    first_stage = list(
        dict.fromkeys(knockout_rows.sort_values("match_id")["stage"])
    )[0]
    fixtures = []
    for _, row in knockout_rows[knockout_rows["stage"] == first_stage].iterrows():
        home = TournamentSimulator._source(
            row["home_source"], group_order, third_assignments, {}, {}
        )
        away = TournamentSimulator._source(
            row["away_source"], group_order, third_assignments, {}, {}
        )
        fixtures.append((int(row["match_id"]), row, home, away))
    return fixtures, group_order


def predict_knockouts(config) -> pd.DataFrame:
    """Predict every fixture in the first knockout round (Round of 32) once the
    group stage is complete, applying the market blend where odds are present.
    """
    results, builder = inputs(config)
    schedule = load_schedule(config["paths"]["schedule"])
    model = WorldCupModel.load(config["paths"]["model"])
    odds_map = load_odds_map(config)
    blend_weight = float(config.get("market", {}).get("blend_weight", 0.0))
    penalty_skill_weight = float(
        config.get("simulation", {}).get("penalty_skill_weight", 0.8927)
    )
    fixtures, _ = resolve_first_knockout_matchups(schedule, results)
    # Team states through the end of the group stage, so knockout predictions
    # reflect group-stage form (results.csv should hold group games only).
    states = builder.build_states(results, before=None)
    rows = []
    for match_id, srow, home, away in fixtures:
        match = srow.copy()
        match["home_team"] = home
        match["away_team"] = away
        match["tournament"] = "FIFA World Cup"
        odds = odds_map.get((srow["date"].date(), home, away))
        if odds is not None:
            match["home_odds"], match["draw_odds"], match["away_odds"] = odds
        features = builder.make_features(match, states)
        prediction = model.predict(features)
        if blend_weight > 0.0:
            blended = WorldCupModel.blend_market(
                (prediction.home_win, prediction.draw, prediction.away_win),
                (
                    features.get("home_market_probability"),
                    features.get("draw_market_probability"),
                    features.get("away_market_probability"),
                ),
                blend_weight,
            )
            prediction.home_win, prediction.draw, prediction.away_win = blended
        # Moneyline: who advances once extra time + penalties resolve the tie,
        # consistent with the (possibly blended) 90-minute split shown above.
        home_rate, away_rate = (value[0] for value in model._rates(
            pd.DataFrame([features])
        ))
        home_advance, away_advance = model.knockout_advance_probabilities(
            float(home_rate),
            float(away_rate),
            float(features["elo_diff"]),
            penalty_skill_weight,
            ninety=(prediction.home_win, prediction.draw, prediction.away_win),
        )
        rows.append(
            {
                "match_id": match_id,
                "date": srow["date"].date().isoformat(),
                "stage": srow["stage"],
                "home_team": home,
                "away_team": away,
                **prediction.__dict__,
                "home_advance": round(home_advance, 3),
                "away_advance": round(away_advance, 3),
            }
        )
    output = pd.DataFrame(rows).round(3)
    path = Path(
        config["paths"].get(
            "knockout_predictions", "outputs/knockout_predictions.csv"
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
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
        match_noise_sigma_scale=config["simulation"].get(
            "match_noise_sigma_scale",
            0.0,
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
    parameters["importance_power"] = config["training"].get(
        "importance_weight_power", 0.75
    )
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
    commands.add_parser("predict-knockouts")
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
    elif args.command == "predict-knockouts":
        print(predict_knockouts(config).to_string(index=False))
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
