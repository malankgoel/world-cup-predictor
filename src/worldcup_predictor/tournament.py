from __future__ import annotations

import itertools
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from .data import HOST_COUNTRY
from .features import FeatureBuilder
from .model import WorldCupModel

STAGE_OUTPUT = {
    "round_of_32": "reach_round_of_32",
    "round_of_16": "reach_round_of_16",
    "quarter_final": "reach_quarter_final",
    "semi_final": "reach_semi_final",
    "final": "reach_final",
}


def assign_third_place_teams(
    sources: list[str], qualified: dict[str, str]
) -> dict[str, str]:
    candidates = {
        source: [group for group in source[1:].split("/") if group in qualified]
        for source in sources
    }
    order = sorted(sources, key=lambda source: (len(candidates[source]), source))

    def search(index: int, used: set[str], assigned: dict[str, str]):
        if index == len(order):
            return assigned
        source = order[index]
        for group in sorted(candidates[source]):
            if group not in used:
                result = search(
                    index + 1,
                    used | {group},
                    {**assigned, source: qualified[group]},
                )
                if result:
                    return result
        return None

    result = search(0, set(), {})
    if result is None:
        raise ValueError("Could not assign qualified third-place teams to bracket")
    return result


def _rank_group(
    teams: list[str], table: dict[str, dict], matches: list[tuple], rng
) -> list[str]:
    def overall(team):
        row = table[team]
        return row["points"], row["gf"] - row["ga"], row["gf"]

    ordered = sorted(teams, key=overall, reverse=True)
    final = []
    index = 0
    while index < len(ordered):
        tied = [ordered[index]]
        key = overall(ordered[index])
        index += 1
        while index < len(ordered) and overall(ordered[index]) == key:
            tied.append(ordered[index])
            index += 1
        if len(tied) > 1:
            mini = {team: {"points": 0, "gf": 0, "ga": 0} for team in tied}
            for home, away, hg, ag in matches:
                if home not in mini or away not in mini:
                    continue
                mini[home]["gf"] += hg
                mini[home]["ga"] += ag
                mini[away]["gf"] += ag
                mini[away]["ga"] += hg
                if hg > ag:
                    mini[home]["points"] += 3
                elif ag > hg:
                    mini[away]["points"] += 3
                else:
                    mini[home]["points"] += 1
                    mini[away]["points"] += 1
            lottery = {team: rng.random() for team in tied}
            tied.sort(
                key=lambda team: (
                    mini[team]["points"],
                    mini[team]["gf"] - mini[team]["ga"],
                    mini[team]["gf"],
                    lottery[team],
                ),
                reverse=True,
            )
        final.extend(tied)
    return final


class TournamentSimulator:
    def __init__(
        self,
        model: WorldCupModel,
        builder: FeatureBuilder,
        results: pd.DataFrame,
        schedule: pd.DataFrame,
        include_tournament: bool = False,
    ):
        self.model = model
        self.builder = builder
        self.schedule = schedule.sort_values(["date", "match_id"])
        tournament_start = self.schedule["date"].min()
        # Team strengths are always frozen at the tournament kickoff so the
        # match-rate baseline never leaks results we are trying to predict.
        # When include_tournament is set (phase 2), already-played tournament
        # games are pinned as observed outcomes instead of being resimulated.
        self.base_states = builder.build_states(results, before=tournament_start)
        if include_tournament:
            tournament_results = results[results["date"] >= tournament_start]
            self.completed = {
                (
                    pd.Timestamp(row.date).date(),
                    row.home_team,
                    row.away_team,
                ): (
                    int(row.home_score),
                    int(row.away_score),
                    row.winner or "",
                )
                for row in tournament_results.itertuples()
            }
        else:
            self.completed = {}
        self.teams = sorted(
            set(self.schedule.loc[self.schedule["stage"] == "group", "home_team"])
            | set(self.schedule.loc[self.schedule["stage"] == "group", "away_team"])
        )
        self.rates = self._build_rate_cache()

    def _build_rate_cache(self):
        keys = []
        feature_rows = []
        for _, row in self.schedule.iterrows():
            pairs = (
                [(row["home_team"], row["away_team"])]
                if row["stage"] == "group"
                else itertools.permutations(self.teams, 2)
            )
            for home, away in pairs:
                match = self._match_row(row, home, away)
                keys.append((int(row["match_id"]), home, away))
                feature_rows.append(
                    self.builder.make_features(match, self.base_states)
                )
        feature_frame = pd.DataFrame(feature_rows)
        home_rates, away_rates = self.model._rates(feature_frame)
        return {
            key: (
                float(home_rate),
                float(away_rate),
                feature["elo_diff"],
                0.5
                * (
                    feature["home_state_uncertainty"]
                    + feature["away_state_uncertainty"]
                ),
            )
            for key, home_rate, away_rate, feature in zip(
                keys, home_rates, away_rates, feature_rows, strict=False
            )
        }

    @staticmethod
    def _match_row(schedule_row, home: str, away: str) -> pd.Series:
        row = schedule_row.copy()
        row["home_team"] = home
        row["away_team"] = away
        row["tournament"] = "FIFA World Cup"
        venue_country = row.get("venue_country", "")
        row["neutral"] = not (
            HOST_COUNTRY.get(home) == venue_country
            or HOST_COUNTRY.get(away) == venue_country
        )
        return row

    def _play(
        self,
        row,
        home: str,
        away: str,
        rng,
        knockout=False,
    ):
        match = self._match_row(row, home, away)
        observed = self.completed.get((match["date"].date(), home, away))
        if observed is None:
            home_rate, away_rate, elo_diff, uncertainty = self.rates[
                (int(row["match_id"]), home, away)
            ]
            sigma = min(0.35, 0.15 * np.sqrt(uncertainty))
            home_rate *= np.exp(
                rng.normal(-0.5 * sigma**2, sigma)
            )
            away_rate *= np.exp(
                rng.normal(-0.5 * sigma**2, sigma)
            )
            home_goals = int(rng.poisson(home_rate))
            away_goals = int(rng.poisson(away_rate))
            if knockout and home_goals == away_goals:
                home_goals += int(rng.poisson(home_rate / 3))
                away_goals += int(rng.poisson(away_rate / 3))
        else:
            home_goals, away_goals, observed_winner = observed
            elo_diff = self.rates[(int(row["match_id"]), home, away)][2]
        if home_goals == away_goals:
            if observed is not None and observed_winner:
                winner = observed_winner
            else:
                penalty_home = 1.0 / (1.0 + np.exp(-elo_diff / 400.0))
                winner = home if rng.random() < penalty_home else away
        else:
            winner = home if home_goals > away_goals else away
        loser = away if winner == home else home
        match["home_score"] = home_goals
        match["away_score"] = away_goals
        return home_goals, away_goals, winner, loser

    @staticmethod
    def _source(
        source: str,
        group_order: dict[str, list[str]],
        third_assignments: dict[str, str],
        winners: dict[int, str],
        losers: dict[int, str],
    ) -> str:
        direct = re.fullmatch(r"([12])([A-L])", source)
        if direct:
            return group_order[direct.group(2)][int(direct.group(1)) - 1]
        if source.startswith("3"):
            return third_assignments[source]
        if source.startswith("W"):
            return winners[int(source[1:])]
        if source.startswith("L"):
            return losers[int(source[1:])]
        return source

    def _one_run(self, rng) -> dict[str, set[str] | str]:
        table = defaultdict(
            lambda: defaultdict(lambda: {"points": 0, "gf": 0, "ga": 0})
        )
        group_matches = defaultdict(list)
        group_rows = self.schedule[self.schedule["stage"] == "group"]
        for _, row in group_rows.iterrows():
            home, away = row["home_team"], row["away_team"]
            hg, ag, _, _ = self._play(row, home, away, rng)
            group = row["group"]
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

        group_order = {}
        for group, values in table.items():
            group_order[group] = _rank_group(
                list(values), values, group_matches[group], rng
            )
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
        qualified_thirds = dict(third_order[:8])
        knockout_rows = self.schedule[self.schedule["stage"] != "group"]
        third_sources = sorted(
            {
                source
                for source in pd.concat(
                    [knockout_rows["home_source"], knockout_rows["away_source"]]
                )
                if str(source).startswith("3")
            }
        )
        third_assignments = assign_third_place_teams(
            third_sources, qualified_thirds
        )
        reached = {column: set() for column in STAGE_OUTPUT.values()}
        for teams in group_order.values():
            reached["reach_round_of_32"].update(teams[:2])
        reached["reach_round_of_32"].update(qualified_thirds.values())
        winners, losers = {}, {}
        for _, row in knockout_rows.iterrows():
            home = self._source(
                row["home_source"], group_order, third_assignments, winners, losers
            )
            away = self._source(
                row["away_source"], group_order, third_assignments, winners, losers
            )
            output_stage = STAGE_OUTPUT.get(row["stage"])
            if output_stage:
                reached[output_stage].update((home, away))
            _, _, winner, loser = self._play(
                row, home, away, rng, knockout=True
            )
            winners[int(row["match_id"])] = winner
            losers[int(row["match_id"])] = loser
        reached["champion"] = winners[104]
        return reached

    def simulate(self, runs: int = 2000, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        counts = {
            team: {**{column: 0 for column in STAGE_OUTPUT.values()}, "champion": 0}
            for team in self.teams
        }
        for _ in range(runs):
            reached = self._one_run(rng)
            for column in STAGE_OUTPUT.values():
                for team in reached[column]:
                    counts[team][column] += 1
            counts[reached["champion"]]["champion"] += 1
        rows = [
            {"team": team, **{key: value / runs for key, value in values.items()}}
            for team, values in counts.items()
        ]
        return pd.DataFrame(rows).sort_values("champion", ascending=False)
