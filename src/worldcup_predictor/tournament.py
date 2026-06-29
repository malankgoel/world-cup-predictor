from __future__ import annotations

import itertools
import re
from collections import defaultdict
from copy import deepcopy

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

# Per-team latent strength, drawn once per simulated tournament and held fixed
# across all of that team's matches in the run. This restores the between-match
# correlation that independent per-match noise washes out: a team that is
# (randomly) stronger than its point estimate stays stronger all tournament,
# which widens advancement/title spreads to realistic levels. Scaled by each
# team's posterior uncertainty, so well-established teams move less. The value
# was selected by 2022 World Cup advancement Brier score.
TEAM_STRENGTH_SCALE = 0.20

# Fraction of the Elo win-probability edge retained in shootouts, fitted on 415
# pre-cutoff international shootouts.
PENALTY_SKILL_WEIGHT = 0.8927

# FIFA's 2026 third-place allocation table assigns the eight advancing
# third-place groups to these bracket slots in this column order.
OFFICIAL_THIRD_PLACE_SOURCE_ORDER = (
    "3C/E/F/H/I",  # vs 1A
    "3E/F/G/I/J",  # vs 1B
    "3B/E/F/I/J",  # vs 1D
    "3A/B/C/D/F",  # vs 1E
    "3A/E/H/I/J",  # vs 1G
    "3C/D/F/G/H",  # vs 1I
    "3D/E/I/J/L",  # vs 1K
    "3E/H/I/J/K",  # vs 1L
)

OFFICIAL_THIRD_PLACE_ASSIGNMENTS = {
    # Current projected advancing third-place groups:
    # 1A-3E, 1B-3J, 1D-3B, 1E-3D, 1G-3I, 1I-3F, 1K-3L, 1L-3K.
    "BDEFIJKL": ("E", "J", "B", "D", "I", "F", "L", "K"),
}


def assign_third_place_teams(
    sources: list[str], qualified: dict[str, str]
) -> dict[str, str]:
    qualified_key = "".join(sorted(qualified))
    official_groups = OFFICIAL_THIRD_PLACE_ASSIGNMENTS.get(qualified_key)
    if official_groups and set(OFFICIAL_THIRD_PLACE_SOURCE_ORDER).issubset(sources):
        assigned = {}
        for source, group in zip(
            OFFICIAL_THIRD_PLACE_SOURCE_ORDER, official_groups, strict=True
        ):
            if group not in source[1:].split("/") or group not in qualified:
                raise ValueError(
                    "Official third-place assignment is incompatible with bracket"
                )
            assigned[source] = qualified[group]
        return assigned

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
        team_strength_scale: float = TEAM_STRENGTH_SCALE,
        penalty_skill_weight: float = PENALTY_SKILL_WEIGHT,
        match_noise_sigma_scale: float = 1.0,
    ):
        self.model = model
        self.builder = builder
        self.schedule = schedule.sort_values(["date", "match_id"])
        self.team_strength_scale = team_strength_scale
        self.penalty_skill_weight = penalty_skill_weight
        self.match_noise_sigma_scale = match_noise_sigma_scale
        tournament_start = self.schedule["date"].min()
        # Team strengths are always frozen at the tournament kickoff so the
        # match-rate baseline never leaks results we are trying to predict.
        # When include_tournament is set (phase 2), already-played tournament
        # games are pinned as observed outcomes instead of being resimulated.
        self.base_states = builder.build_states(results, before=tournament_start)
        if include_tournament:
            tournament_results = results[results["date"] >= tournament_start]
            # Key both home/away orderings: the schedule and the results feed can
            # disagree on which side is "home" for neutral/host games (e.g. the
            # feed lists Canada vs Switzerland, the schedule Switzerland vs
            # Canada). Storing the swapped orientation too means the lookup in
            # _play pins the real result instead of resimulating the match.
            self.completed = {}
            for row in tournament_results.itertuples():
                played_on = pd.Timestamp(row.date).date()
                home_goals = int(row.home_score)
                away_goals = int(row.away_score)
                winner = row.winner or ""
                self.completed[(played_on, row.home_team, row.away_team)] = (
                    home_goals,
                    away_goals,
                    winner,
                )
                self.completed[(played_on, row.away_team, row.home_team)] = (
                    away_goals,
                    home_goals,
                    winner,
                )
        else:
            self.completed = {}
        self.teams = sorted(
            set(self.schedule.loc[self.schedule["stage"] == "group", "home_team"])
            | set(self.schedule.loc[self.schedule["stage"] == "group", "away_team"])
        )
        # Derive the bracket shape from the schedule rather than hardwiring the
        # 2026 layout, so the same engine handles the 2022 (32-team) format too.
        knockout = self.schedule[self.schedule["stage"] != "group"]
        ordered_stages = list(
            dict.fromkeys(knockout.sort_values("match_id")["stage"])
        )
        # Reaching the third-place match is a consolation, not progress.
        self.stage_outputs = {
            stage: f"reach_{stage}"
            for stage in ordered_stages
            if stage != "third_place"
        }
        self.final_match_id = int(
            self.schedule.loc[self.schedule["stage"] == "final", "match_id"].max()
        )
        # The first knockout round is what every group qualifier "reaches".
        self.first_knockout_stage = ordered_stages[0] if ordered_stages else None
        self.rates = self._build_rate_cache()

    def _build_rate_cache(self):
        keys = []
        feature_rows = []
        context_states = deepcopy(self.base_states)
        group_rows = self.schedule[self.schedule["stage"] == "group"]
        for _, row in group_rows.iterrows():
            home, away = row["home_team"], row["away_team"]
            match = self._match_row(row, home, away)
            keys.append((int(row["match_id"]), home, away))
            feature_rows.append(self.builder.make_features(match, context_states))
            self.builder.apply_fixture_context(match, context_states)
        knockout_rows = self.schedule[self.schedule["stage"] != "group"]
        for _, row in knockout_rows.iterrows():
            for home, away in itertools.permutations(self.teams, 2):
                match = self._match_row(row, home, away)
                keys.append((int(row["match_id"]), home, away))
                feature_rows.append(
                    self.builder.make_features(match, context_states)
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

    @staticmethod
    def _apply_team_shock(
        home_rate: float,
        away_rate: float,
        home_shock: tuple[float, float],
        away_shock: tuple[float, float],
    ) -> tuple[float, float]:
        """Apply a zero-sum, per-run relative-strength tilt to both rates.

        The correction term keeps each rate's expectation unbiased over runs,
        and a shared shock for both teams leaves the matchup unchanged.
        """
        home_z, home_sigma = home_shock
        away_z, away_sigma = away_shock
        correction = 0.5 * (home_sigma**2 + away_sigma**2)
        home_rate = home_rate * np.exp((home_z - away_z) - correction)
        away_rate = away_rate * np.exp((away_z - home_z) - correction)
        return home_rate, away_rate

    @staticmethod
    def _shootout_probability(
        elo_diff: float,
        skill_weight: float = PENALTY_SKILL_WEIGHT,
    ) -> float:
        """Home win probability in a shootout, shrunk toward 0.5 from Elo."""
        elo_home = 1.0 / (1.0 + np.exp(-elo_diff / 400.0))
        return 0.5 + skill_weight * (elo_home - 0.5)

    def _sample_score(self, home_rate: float, away_rate: float, rng) -> tuple[int, int]:
        matrix = self.model.outcome_probabilities(home_rate, away_rate)[3]
        flat_index = int(rng.choice(matrix.size, p=matrix.ravel()))
        home_goals, away_goals = np.unravel_index(flat_index, matrix.shape)
        return int(home_goals), int(away_goals)

    def _play(
        self,
        row,
        home: str,
        away: str,
        rng,
        knockout=False,
        shock: dict[str, tuple[float, float]] | None = None,
    ):
        match = self._match_row(row, home, away)
        observed = self.completed.get((match["date"].date(), home, away))
        if observed is None:
            home_rate, away_rate, elo_diff, uncertainty = self.rates[
                (int(row["match_id"]), home, away)
            ]
            # The score is already sampled from the Poisson/NB matrix below, so
            # this per-match rate shock is supplementary; scaled by
            # match_noise_sigma_scale (default 0.0) to avoid double-counting
            # match variance on top of the per-tournament team shock, which used
            # to wash out favorites' advancement odds.
            sigma = self.match_noise_sigma_scale * min(0.35, 0.15 * np.sqrt(uncertainty))
            if sigma > 0.0:
                home_rate *= np.exp(rng.normal(-0.5 * sigma**2, sigma))
                away_rate *= np.exp(rng.normal(-0.5 * sigma**2, sigma))
            if shock is not None:
                home_rate, away_rate = self._apply_team_shock(
                    home_rate, away_rate, shock[home], shock[away]
                )
            home_goals, away_goals = self._sample_score(
                home_rate,
                away_rate,
                rng,
            )
            if knockout and home_goals == away_goals:
                extra_home, extra_away = self._sample_score(
                    home_rate / 3,
                    away_rate / 3,
                    rng,
                )
                home_goals += extra_home
                away_goals += extra_away
        else:
            home_goals, away_goals, observed_winner = observed
            elo_diff = self.rates[(int(row["match_id"]), home, away)][2]
        if home_goals == away_goals:
            if observed is not None and observed_winner:
                winner = observed_winner
            else:
                penalty_home = self._shootout_probability(
                    elo_diff,
                    self.penalty_skill_weight,
                )
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
        # One latent strength draw per team for this whole tournament run.
        shock: dict[str, tuple[float, float]] = {}
        for team in self.teams:
            state = self.base_states.get(team)
            variance = (
                state.attack_var + state.defense_var if state is not None else 0.70
            )
            sigma = self.team_strength_scale * float(np.sqrt(variance))
            shock[team] = (float(rng.normal(0.0, sigma)), sigma)
        table = defaultdict(
            lambda: defaultdict(lambda: {"points": 0, "gf": 0, "ga": 0})
        )
        group_matches = defaultdict(list)
        group_rows = self.schedule[self.schedule["stage"] == "group"]
        for _, row in group_rows.iterrows():
            home, away = row["home_team"], row["away_team"]
            hg, ag, _, _ = self._play(row, home, away, rng, shock=shock)
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
        # Best third-placed teams only exist in formats that use them (e.g.
        # 2026). With top-two-only brackets (2022) this block is skipped.
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
            qualified_thirds = dict(third_order[: len(third_sources)])
            third_assignments = assign_third_place_teams(
                third_sources, qualified_thirds
            )
        else:
            third_assignments = {}

        # The first knockout round's participants (every group qualifier) are
        # added when those matches are played, so no manual seeding is needed.
        reached = {column: set() for column in self.stage_outputs.values()}
        winners, losers = {}, {}
        for _, row in knockout_rows.iterrows():
            home = self._source(
                row["home_source"], group_order, third_assignments, winners, losers
            )
            away = self._source(
                row["away_source"], group_order, third_assignments, winners, losers
            )
            output_stage = self.stage_outputs.get(row["stage"])
            if output_stage:
                reached[output_stage].update((home, away))
            _, _, winner, loser = self._play(
                row, home, away, rng, knockout=True, shock=shock
            )
            winners[int(row["match_id"])] = winner
            losers[int(row["match_id"])] = loser
        reached["champion"] = winners[self.final_match_id]
        return reached

    def simulate(self, runs: int = 2000, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        counts = {
            team: {**{column: 0 for column in self.stage_outputs.values()}, "champion": 0}
            for team in self.teams
        }
        for _ in range(runs):
            reached = self._one_run(rng)
            for column in self.stage_outputs.values():
                for team in reached[column]:
                    counts[team][column] += 1
            counts[reached["champion"]]["champion"] += 1
        rows = [
            {"team": team, **{key: value / runs for key, value in values.items()}}
            for team, values in counts.items()
        ]
        return pd.DataFrame(rows).sort_values("champion", ascending=False)
