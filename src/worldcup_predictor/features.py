from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from math import asin, cos, log1p, radians, sin, sqrt

import numpy as np
import pandas as pd

from .data import CONFEDERATION, HOST_COUNTRY, VENUE_ATTRIBUTES, normalize_team

FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form_points",
    "away_form_points",
    "home_form_gf",
    "away_form_gf",
    "home_form_ga",
    "away_form_ga",
    "home_form_xgf",
    "away_form_xgf",
    "home_form_xga",
    "away_form_xga",
    "home_rest_days",
    "away_rest_days",
    "home_travel_km",
    "away_travel_km",
    "venue_altitude_m",
    "venue_temperature_c",
    "venue_humidity",
    "experience_diff",
    "home_attack_state",
    "away_attack_state",
    "attack_state_diff",
    "home_defense_state",
    "away_defense_state",
    "defense_state_diff",
    "home_state_uncertainty",
    "away_state_uncertainty",
    "home_advantage",
    "away_advantage",
    "neutral",
    "importance",
    "is_knockout",
    "home_market_probability",
    "draw_market_probability",
    "away_market_probability",
    "home_rank_points",
    "away_rank_points",
    "rank_points_diff",
    "home_xi_rating",
    "away_xi_rating",
    "xi_rating_diff",
    "home_depth_rating",
    "away_depth_rating",
    "home_star_rating",
    "away_star_rating",
    "star_rating_diff",
    "home_top3_rating",
    "away_top3_rating",
    "top3_rating_diff",
    "home_squad_attack",
    "away_squad_attack",
    "squad_attack_diff",
    "home_squad_defense",
    "away_squad_defense",
    "squad_defense_diff",
    "home_talent_uncertainty",
    "away_talent_uncertainty",
    "home_chemistry",
    "away_chemistry",
    "chemistry_diff",
    "home_max_same_club",
    "away_max_same_club",
    "home_avg_age",
    "away_avg_age",
    "same_confederation",
]

for _confederation in ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA"):
    FEATURE_COLUMNS.extend(
        [f"home_confed_{_confederation.lower()}", f"away_confed_{_confederation.lower()}"]
    )

POSITION_WEIGHTS = {
    "GK": (0.0, 1.0),
    "DEF": (0.2, 1.0),
    "MID": (0.6, 0.5),
    "FWD": (1.0, 0.2),
}

# Baseline scoring rates for the latent-state update. Historically this used a
# fixed home/away intercept of log(1.25)/log(1.10); that asymmetry doubled as a
# second home-advantage term and leaked into neutral matches, where the "home"
# label is arbitrary. We instead use a symmetric neutral baseline plus a single
# advantage tilt, chosen so the rates are IDENTICAL to the old ones on
# non-neutral matches (home_adv=1, away_adv=0) while staying symmetric when the
# venue is neutral (home_adv=away_adv=0).
_LOG_125 = float(np.log(1.25))
_LOG_110 = float(np.log(1.10))
NEUTRAL_BASE_LOG = 0.5 * (_LOG_125 + _LOG_110)
HOME_ADVANTAGE_LOG = 0.5 * (_LOG_125 - _LOG_110) + 0.15

DEFAULT_STATE_PARAMETERS = {
    "neutral_base_log": NEUTRAL_BASE_LOG,
    "home_advantage_log": HOME_ADVANTAGE_LOG,
    "observation_variance": 1.5,
    "update_step": 0.5,
    "surprise_clip": 1.2,
    "state_mean_clip": 1.5,
    "variance_drift_per_day": 0.0005,
    "variance_ceiling": 0.60,
}

DEFAULT_ELO_PARAMETERS = {
    "k_factor": 28.0,
    "home_advantage_points": 80.0,
}


def match_importance(tournament: str) -> float:
    name = str(tournament).lower()
    if "world cup" in name and "qualification" not in name:
        return 1.0
    if any(word in name for word in ("euro", "copa am", "african cup", "asian cup")):
        return 0.85
    if "qualification" in name or "qualifier" in name:
        return 0.70
    if "nations league" in name:
        return 0.55
    if "friendly" in name:
        return 0.25
    return 0.50


@dataclass
class TeamState:
    elo: float = 1500.0
    attack_mean: float = 0.0
    attack_var: float = 0.35
    defense_mean: float = 0.0
    defense_var: float = 0.35
    played: int = 0
    last_date: pd.Timestamp | None = None
    history: list[tuple[float, float, float, float, float]] = field(
        default_factory=list
    )
    last_latitude: float | None = None
    last_longitude: float | None = None

    def form(self, window: int) -> tuple[float, float, float, float, float]:
        recent = self.history[-window:]
        if not recent:
            return 1.0, 1.2, 1.2, 1.2, 1.2
        weights = np.power(0.82, np.arange(len(recent) - 1, -1, -1))
        values = np.asarray(recent, dtype=float)
        means = np.average(values, axis=0, weights=weights)
        return tuple(float(value) for value in means)


class RankingLookup:
    def __init__(self, rankings: pd.DataFrame):
        self.values: dict[str, tuple[list[pd.Timestamp], list[float]]] = {}
        if rankings.empty:
            return
        frame = rankings.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["team"] = frame["team"].map(normalize_team)
        frame["points"] = pd.to_numeric(frame["points"], errors="coerce")
        for team, group in frame.dropna(subset=["points"]).groupby("team"):
            group = group.sort_values("date")
            self.values[team] = (
                group["date"].tolist(),
                group["points"].astype(float).tolist(),
            )

    def get(self, team: str, date: pd.Timestamp) -> float:
        dates, values = self.values.get(team, ([], []))
        index = bisect_right(dates, date) - 1
        return values[index] if index >= 0 else np.nan


class SquadLookup:
    def __init__(self, squads: pd.DataFrame):
        self.snapshots: dict[str, list[tuple[pd.Timestamp, dict[str, float]]]] = {}
        if squads.empty:
            return
        frame = squads.copy()
        frame["as_of"] = pd.to_datetime(frame["as_of"])
        frame["team"] = frame["team"].map(normalize_team)
        frame["talent"] = pd.to_numeric(frame["talent"], errors="coerce")
        frame["talent_se"] = pd.to_numeric(frame["talent_se"], errors="coerce")
        frame["exp_minutes"] = pd.to_numeric(
            frame["exp_minutes"], errors="coerce"
        ).clip(0, 1)
        frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
        frame["is_starter"] = (
            frame["is_starter"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin({"1", "true", "yes", "y"})
        )
        frame["available"] = (
            frame["available"]
            .fillna(True)
            .astype(str)
            .str.lower()
            .isin({"1", "true", "yes", "y"})
        )
        position = frame["position"].astype(str).str.upper()
        frame["position"] = np.select(
            [
                position.isin({"GK", "G"}),
                position.isin({"DEF", "D", "CB", "LB", "RB", "LWB", "RWB"}),
                position.isin({"MID", "M", "MF", "CM", "CAM", "CDM", "LM", "RM"}),
                position.isin({"FWD", "F", "FW", "ST", "CF", "LW", "RW"}),
            ],
            ["GK", "DEF", "MID", "FWD"],
            default=position,
        )
        default_minutes = np.where(frame["is_starter"], 0.9, 0.25)
        frame["exp_minutes"] = frame["exp_minutes"].fillna(
            pd.Series(default_minutes, index=frame.index)
        )
        frame["talent_se"] = frame["talent_se"].fillna(3.0)
        for (team, as_of), group in frame.groupby(["team", "as_of"]):
            features = self._summarize(group)
            self.snapshots.setdefault(team, []).append((as_of, features))
        for values in self.snapshots.values():
            values.sort(key=lambda item: item[0])

    @staticmethod
    def _summarize(group: pd.DataFrame) -> dict[str, float]:
        rated = group.dropna(subset=["talent"]).copy()
        rated["minutes_weight"] = rated["exp_minutes"] * rated["available"]
        rated = rated.sort_values(
            ["minutes_weight", "talent"], ascending=False
        )
        available = rated[rated["available"]]
        starters = available[available["is_starter"]]
        if len(starters) < 11:
            starters = pd.concat(
                [starters, available[~available.index.isin(starters.index)]]
            ).head(11)
        else:
            starters = starters.head(11)
        clubs = starters["club"].replace("", np.nan).dropna().value_counts()
        pairs = float(sum(count * (count - 1) / 2 for count in clubs))
        att_weights = rated.apply(
            lambda row: row["minutes_weight"]
            * POSITION_WEIGHTS.get(row["position"], (0.4, 0.4))[0],
            axis=1,
        )
        def_weights = rated.apply(
            lambda row: row["minutes_weight"]
            * POSITION_WEIGHTS.get(row["position"], (0.4, 0.4))[1],
            axis=1,
        )

        def weighted_mean(values, weights):
            return (
                float(np.average(values, weights=weights))
                if weights.sum() > 0
                else np.nan
            )

        attack = weighted_mean(rated["talent"], att_weights)
        outfield = rated[rated["position"] != "GK"]
        outfield_weights = def_weights.loc[outfield.index]
        outfield_defense = weighted_mean(outfield["talent"], outfield_weights)
        goalkeepers = rated[
            (rated["position"] == "GK") & (rated["minutes_weight"] > 0)
        ]
        goalkeeper = (
            float(goalkeepers.iloc[0]["talent"]) if len(goalkeepers) else np.nan
        )
        defense = (
            0.75 * outfield_defense + 0.25 * goalkeeper
            if np.isfinite(outfield_defense) and np.isfinite(goalkeeper)
            else outfield_defense
        )
        uncertainty = weighted_mean(rated["talent_se"], rated["minutes_weight"])
        return {
            "xi_rating": float(starters["talent"].mean()),
            "depth_rating": float(available.head(23)["talent"].mean()),
            "star_rating": float(available["talent"].max()),
            "top3_rating": float(available.nlargest(3, "talent")["talent"].mean()),
            "squad_attack": attack,
            "squad_defense": defense,
            "talent_uncertainty": uncertainty,
            "chemistry": pairs / 55.0,
            "max_same_club": float(clubs.max()) if len(clubs) else np.nan,
            "avg_age": float(starters["age"].mean()),
        }

    def get(self, team: str, date: pd.Timestamp) -> dict[str, float]:
        snapshots = self.snapshots.get(team, [])
        dates = [item[0] for item in snapshots]
        index = bisect_right(dates, date) - 1
        if index < 0:
            return {
                "xi_rating": np.nan,
                "depth_rating": np.nan,
                "star_rating": np.nan,
                "top3_rating": np.nan,
                "squad_attack": np.nan,
                "squad_defense": np.nan,
                "talent_uncertainty": np.nan,
                "chemistry": np.nan,
                "max_same_club": np.nan,
                "avg_age": np.nan,
            }
        return snapshots[index][1]


class FeatureBuilder:
    def __init__(
        self,
        rankings: pd.DataFrame,
        squads: pd.DataFrame,
        form_matches: int = 8,
        state_parameters: dict | None = None,
        elo_parameters: dict | None = None,
    ):
        self.rankings = RankingLookup(rankings)
        self.squads = SquadLookup(squads)
        self.form_matches = form_matches
        self.state_parameters = {
            **DEFAULT_STATE_PARAMETERS,
            **(state_parameters or {}),
        }
        self.elo_parameters = {
            **DEFAULT_ELO_PARAMETERS,
            **(elo_parameters or {}),
        }

    @staticmethod
    def _state(states: dict[str, TeamState], team: str) -> TeamState:
        return states.setdefault(team, TeamState())

    @staticmethod
    def _advantages(match: pd.Series, home: str, away: str) -> tuple[float, float]:
        venue_country = str(match.get("venue_country", match.get("country", "")))
        neutral = bool(match.get("neutral", True))
        if neutral:
            return 0.0, 0.0
        if venue_country and HOST_COUNTRY.get(away) == venue_country:
            return 0.0, 1.0
        return 1.0, 0.0

    @staticmethod
    def _venue(match: pd.Series) -> tuple[float, float, float, float, float]:
        name = str(match.get("venue", match.get("city", ""))).strip()
        return VENUE_ATTRIBUTES.get(
            name,
            (np.nan, np.nan, np.nan, np.nan, np.nan),
        )

    @staticmethod
    def _distance_km(
        latitude_1: float | None,
        longitude_1: float | None,
        latitude_2: float,
        longitude_2: float,
    ) -> float:
        if (
            latitude_1 is None
            or longitude_1 is None
            or not np.isfinite(latitude_2)
            or not np.isfinite(longitude_2)
        ):
            return np.nan
        lat1, lon1, lat2, lon2 = map(
            radians,
            (latitude_1, longitude_1, latitude_2, longitude_2),
        )
        value = (
            sin((lat2 - lat1) / 2) ** 2
            + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
        )
        return float(2 * 6371.0 * asin(sqrt(value)))

    @staticmethod
    def _market_probabilities(match: pd.Series) -> tuple[float, float, float]:
        odds = np.asarray(
            [
                pd.to_numeric(match.get("home_odds"), errors="coerce"),
                pd.to_numeric(match.get("draw_odds"), errors="coerce"),
                pd.to_numeric(match.get("away_odds"), errors="coerce"),
            ],
            dtype=float,
        )
        if not np.isfinite(odds).all() or np.any(odds <= 1.0):
            return np.nan, np.nan, np.nan
        inverse = 1.0 / odds
        probabilities = inverse / inverse.sum()
        return tuple(float(value) for value in probabilities)

    @staticmethod
    def _is_knockout(match: pd.Series) -> float:
        raw_stage = match.get("stage", "")
        stage = "" if pd.isna(raw_stage) else str(raw_stage).strip().lower()
        return float(stage not in {"", "group", "group_stage", "league"})

    def make_features(
        self, match: pd.Series, states: dict[str, TeamState]
    ) -> dict[str, float]:
        date = pd.Timestamp(match["date"])
        home = normalize_team(match["home_team"])
        away = normalize_team(match["away_team"])
        home_state = self._state(states, home)
        away_state = self._state(states, away)
        hp, hgf, hga, hxgf, hxga = home_state.form(self.form_matches)
        ap, agf, aga, axgf, axga = away_state.form(self.form_matches)
        home_rank = self.rankings.get(home, date)
        away_rank = self.rankings.get(away, date)
        home_squad = self.squads.get(home, date)
        away_squad = self.squads.get(away, date)
        home_adv, away_adv = self._advantages(match, home, away)
        latitude, longitude, altitude, temperature, humidity = self._venue(match)
        market_home, market_draw, market_away = self._market_probabilities(match)

        def rest_days(state: TeamState) -> float:
            if state.last_date is None:
                return 30.0
            return float(np.clip((date - state.last_date).days, 0, 90))

        def current_var(state: TeamState, name: str) -> float:
            drift_days = rest_days(state)
            return min(
                self.state_parameters["variance_ceiling"],
                getattr(state, name)
                + self.state_parameters["variance_drift_per_day"] * drift_days,
            )

        features = {
            "home_elo": home_state.elo,
            "away_elo": away_state.elo,
            "elo_diff": home_state.elo - away_state.elo,
            "home_form_points": hp,
            "away_form_points": ap,
            "home_form_gf": hgf,
            "away_form_gf": agf,
            "home_form_ga": hga,
            "away_form_ga": aga,
            "home_form_xgf": hxgf,
            "away_form_xgf": axgf,
            "home_form_xga": hxga,
            "away_form_xga": axga,
            "home_rest_days": rest_days(home_state),
            "away_rest_days": rest_days(away_state),
            "home_travel_km": self._distance_km(
                home_state.last_latitude,
                home_state.last_longitude,
                latitude,
                longitude,
            ),
            "away_travel_km": self._distance_km(
                away_state.last_latitude,
                away_state.last_longitude,
                latitude,
                longitude,
            ),
            "venue_altitude_m": altitude,
            "venue_temperature_c": temperature,
            "venue_humidity": humidity,
            "experience_diff": log1p(home_state.played) - log1p(away_state.played),
            "home_attack_state": home_state.attack_mean,
            "away_attack_state": away_state.attack_mean,
            "attack_state_diff": home_state.attack_mean - away_state.attack_mean,
            "home_defense_state": home_state.defense_mean,
            "away_defense_state": away_state.defense_mean,
            "defense_state_diff": home_state.defense_mean - away_state.defense_mean,
            "home_state_uncertainty": current_var(
                home_state, "attack_var"
            ) + current_var(home_state, "defense_var"),
            "away_state_uncertainty": current_var(
                away_state, "attack_var"
            ) + current_var(away_state, "defense_var"),
            "home_advantage": home_adv,
            "away_advantage": away_adv,
            "neutral": float(bool(match.get("neutral", True))),
            "importance": match_importance(match.get("tournament", "World Cup")),
            "is_knockout": self._is_knockout(match),
            "home_market_probability": market_home,
            "draw_market_probability": market_draw,
            "away_market_probability": market_away,
            "home_rank_points": home_rank,
            "away_rank_points": away_rank,
            "rank_points_diff": home_rank - away_rank,
        }
        for key in (
            "xi_rating",
            "depth_rating",
            "star_rating",
            "top3_rating",
            "squad_attack",
            "squad_defense",
            "talent_uncertainty",
            "chemistry",
            "max_same_club",
            "avg_age",
        ):
            features[f"home_{key}"] = home_squad[key]
            features[f"away_{key}"] = away_squad[key]
        features["xi_rating_diff"] = (
            home_squad["xi_rating"] - away_squad["xi_rating"]
        )
        features["star_rating_diff"] = (
            home_squad["star_rating"] - away_squad["star_rating"]
        )
        features["top3_rating_diff"] = (
            home_squad["top3_rating"] - away_squad["top3_rating"]
        )
        features["chemistry_diff"] = (
            home_squad["chemistry"] - away_squad["chemistry"]
        )
        features["squad_attack_diff"] = (
            home_squad["squad_attack"] - away_squad["squad_attack"]
        )
        features["squad_defense_diff"] = (
            home_squad["squad_defense"] - away_squad["squad_defense"]
        )
        home_confed = CONFEDERATION.get(home)
        away_confed = CONFEDERATION.get(away)
        features["same_confederation"] = float(
            home_confed is not None and home_confed == away_confed
        )
        for confederation in ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA"):
            key = confederation.lower()
            features[f"home_confed_{key}"] = float(home_confed == confederation)
            features[f"away_confed_{key}"] = float(away_confed == confederation)
        return features

    def apply_result(
        self, match: pd.Series, states: dict[str, TeamState]
    ) -> None:
        date = pd.Timestamp(match["date"])
        home = normalize_team(match["home_team"])
        away = normalize_team(match["away_team"])
        home_score = float(match["home_score"])
        away_score = float(match["away_score"])
        home_state = self._state(states, home)
        away_state = self._state(states, away)
        home_adv, away_adv = self._advantages(match, home, away)
        state_parameters = self.state_parameters
        for state in (home_state, away_state):
            if state.last_date is not None:
                drift = state_parameters["variance_drift_per_day"] * min(
                    (date - state.last_date).days,
                    90,
                )
                state.attack_var = min(
                    state_parameters["variance_ceiling"],
                    state.attack_var + drift,
                )
                state.defense_var = min(
                    state_parameters["variance_ceiling"],
                    state.defense_var + drift,
                )
        home_rate = np.exp(
            state_parameters["neutral_base_log"]
            + home_state.attack_mean
            - away_state.defense_mean
            + state_parameters["home_advantage_log"] * (home_adv - away_adv)
        )
        away_rate = np.exp(
            state_parameters["neutral_base_log"]
            + away_state.attack_mean
            - home_state.defense_mean
            + state_parameters["home_advantage_log"] * (away_adv - home_adv)
        )

        def bayes_update(
            attack: TeamState,
            defense: TeamState,
            goals: float,
            expected: float,
        ) -> None:
            surprise = float(
                np.clip(
                    np.log((goals + 0.5) / (expected + 0.5)),
                    -state_parameters["surprise_clip"],
                    state_parameters["surprise_clip"],
                )
            )
            observation_variance = state_parameters["observation_variance"]
            attack_gain = attack.attack_var / (
                attack.attack_var + observation_variance
            )
            defense_gain = defense.defense_var / (
                defense.defense_var + observation_variance
            )
            step = state_parameters["update_step"]
            attack.attack_mean += step * attack_gain * surprise
            defense.defense_mean -= step * defense_gain * surprise
            mean_clip = state_parameters["state_mean_clip"]
            attack.attack_mean = float(
                np.clip(attack.attack_mean, -mean_clip, mean_clip)
            )
            defense.defense_mean = float(
                np.clip(defense.defense_mean, -mean_clip, mean_clip)
            )
            attack.attack_var = (1 - attack_gain) * attack.attack_var
            defense.defense_var = (1 - defense_gain) * defense.defense_var

        bayes_update(home_state, away_state, home_score, home_rate)
        bayes_update(away_state, home_state, away_score, away_rate)
        adjusted_home = home_state.elo + self.elo_parameters[
            "home_advantage_points"
        ] * (home_adv - away_adv)
        expected_home = 1.0 / (1.0 + 10 ** ((away_state.elo - adjusted_home) / 400))
        actual_home = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
        margin = max(1.0, np.log1p(abs(home_score - away_score)))
        k = (
            self.elo_parameters["k_factor"]
            * match_importance(match.get("tournament", ""))
            * margin
        )
        change = k * (actual_home - expected_home)
        home_state.elo += change
        away_state.elo -= change
        home_points = 3.0 if home_score > away_score else 1.0 if home_score == away_score else 0.0
        away_points = 3.0 if away_score > home_score else 1.0 if home_score == away_score else 0.0
        home_xg = pd.to_numeric(match.get("home_xg"), errors="coerce")
        away_xg = pd.to_numeric(match.get("away_xg"), errors="coerce")
        if not np.isfinite(home_xg):
            home_xg = home_score
        if not np.isfinite(away_xg):
            away_xg = away_score
        home_state.history.append(
            (home_points, home_score, away_score, float(home_xg), float(away_xg))
        )
        away_state.history.append(
            (away_points, away_score, home_score, float(away_xg), float(home_xg))
        )
        home_state.played += 1
        away_state.played += 1
        home_state.last_date = date
        away_state.last_date = date
        latitude, longitude, *_ = self._venue(match)
        if np.isfinite(latitude) and np.isfinite(longitude):
            home_state.last_latitude = latitude
            home_state.last_longitude = longitude
            away_state.last_latitude = latitude
            away_state.last_longitude = longitude

    def apply_fixture_context(
        self,
        match: pd.Series,
        states: dict[str, TeamState],
    ) -> None:
        """Advance rest/travel context for a future fixture without using a result."""
        date = pd.Timestamp(match["date"])
        latitude, longitude, *_ = self._venue(match)
        for team in (match["home_team"], match["away_team"]):
            state = self._state(states, normalize_team(team))
            state.last_date = date
            if np.isfinite(latitude) and np.isfinite(longitude):
                state.last_latitude = latitude
                state.last_longitude = longitude

    @staticmethod
    def _infer_stage_context(results: pd.DataFrame) -> pd.DataFrame:
        frame = results.copy()
        if "stage" not in frame:
            frame["stage"] = ""
        world_cup = frame["tournament"].astype(str).eq("FIFA World Cup")
        for _, group in frame[world_cup].groupby(frame.loc[world_cup, "date"].dt.year):
            # Every men's World Cup since 1986 has had a 16-team knockout phase.
            # The project trains from 2000, so the final 16 matches are stable.
            if len(group) >= 64:
                frame.loc[group.sort_values("date").tail(16).index, "stage"] = "knockout"
                remaining = group.index.difference(
                    group.sort_values("date").tail(16).index
                )
                frame.loc[remaining, "stage"] = "group"
        return frame

    def training_frame(self, results: pd.DataFrame) -> pd.DataFrame:
        states: dict[str, TeamState] = {}
        rows = []
        ordered = self._infer_stage_context(results.sort_values("date"))
        for _, match in ordered.iterrows():
            row = self.make_features(match, states)
            row.update(
                {
                    "date": match["date"],
                    "home_team": match["home_team"],
                    "away_team": match["away_team"],
                    "home_score": match["home_score"],
                    "away_score": match["away_score"],
                }
            )
            rows.append(row)
            self.apply_result(match, states)
        return pd.DataFrame(rows)

    def build_states(
        self, results: pd.DataFrame, before: pd.Timestamp | None = None
    ) -> dict[str, TeamState]:
        states: dict[str, TeamState] = {}
        for _, match in results.sort_values("date").iterrows():
            if before is not None and pd.Timestamp(match["date"]) >= before:
                break
            self.apply_result(match, states)
        return states
