import numpy as np
import pandas as pd

from worldcup_predictor.features import (
    FeatureBuilder,
    RankingLookup,
    SquadLookup,
)


def test_features_are_created_before_result_is_applied():
    results = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-01"),
                "home_team": "A",
                "away_team": "B",
                "home_score": 3,
                "away_score": 0,
                "tournament": "Friendly",
                "country": "A",
                "neutral": False,
            },
            {
                "date": pd.Timestamp("2025-02-01"),
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 1,
                "tournament": "Friendly",
                "country": "A",
                "neutral": False,
            },
        ]
    )
    frame = FeatureBuilder(pd.DataFrame(), pd.DataFrame()).training_frame(results)
    assert frame.loc[0, "elo_diff"] == 0
    assert frame.loc[1, "elo_diff"] > 0
    assert frame.loc[0, "home_form_gf"] == 1.2
    assert frame.loc[1, "home_form_gf"] == 3


def test_squad_features_measure_same_club_chemistry():
    squad = pd.DataFrame(
        [
            {
                "as_of": "2026-01-01",
                "team": "A",
                "player": f"P{i}",
                "club": "Club 1" if i < 4 else f"Club {i}",
                "position": "MF",
                "talent": 80 + i / 10,
                "talent_se": 2,
                "exp_minutes": 0.9,
                "available": True,
                "age": 25,
                "is_starter": True,
            }
            for i in range(11)
        ]
    )
    builder = FeatureBuilder(pd.DataFrame(), squad)
    match = pd.Series(
        {
            "date": pd.Timestamp("2026-06-01"),
            "home_team": "A",
            "away_team": "B",
            "neutral": True,
            "tournament": "World Cup",
        }
    )
    features = builder.make_features(match, {})
    assert features["home_max_same_club"] == 4
    assert features["home_chemistry"] == 6 / 55
    assert features["home_squad_attack"] > 80


def test_result_updates_bayesian_attack_and_defense_state():
    builder = FeatureBuilder(pd.DataFrame(), pd.DataFrame())
    states = {}
    match = pd.Series(
        {
            "date": pd.Timestamp("2026-06-01"),
            "home_team": "A",
            "away_team": "B",
            "home_score": 5,
            "away_score": 0,
            "neutral": True,
            "tournament": "World Cup",
        }
    )
    builder.apply_result(match, states)
    assert states["A"].attack_mean > 0
    assert states["A"].defense_mean > 0
    assert states["B"].attack_mean < 0
    assert states["B"].defense_mean < 0
    assert states["A"].attack_var < 0.35


def _series_of_matches():
    rows = []
    for month, (hs, asc) in enumerate([(2, 0), (1, 1), (0, 3)], start=1):
        rows.append(
            {
                "date": pd.Timestamp(f"2025-0{month}-01"),
                "home_team": "A",
                "away_team": "B",
                "home_score": hs,
                "away_score": asc,
                "tournament": "Friendly",
                "country": "A",
                "neutral": False,
            }
        )
    return pd.DataFrame(rows)


def test_no_future_result_leaks_into_earlier_features():
    base = _series_of_matches()
    first = FeatureBuilder(pd.DataFrame(), pd.DataFrame()).training_frame(base)

    altered = base.copy()
    altered.loc[2, ["home_score", "away_score"]] = [9, 0]
    second = FeatureBuilder(pd.DataFrame(), pd.DataFrame()).training_frame(altered)

    label_columns = {"date", "home_team", "away_team", "home_score", "away_score"}
    feature_columns = [c for c in first.columns if c not in label_columns]
    # Rewriting the LAST match must not change features of the earlier matches.
    pd.testing.assert_frame_equal(
        first.loc[:1, feature_columns], second.loc[:1, feature_columns]
    )


def test_ranking_lookup_returns_latest_snapshot_on_or_before_date():
    rankings = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-06-01"],
            "team": ["A", "A"],
            "rank": [10, 8],
            "points": [1500.0, 1550.0],
        }
    )
    lookup = RankingLookup(rankings)
    assert lookup.get("A", pd.Timestamp("2025-03-01")) == 1500.0
    assert lookup.get("A", pd.Timestamp("2025-07-01")) == 1550.0
    assert np.isnan(lookup.get("A", pd.Timestamp("2024-01-01")))
    assert np.isnan(lookup.get("Unknown", pd.Timestamp("2025-07-01")))


def test_squad_lookup_returns_latest_snapshot_on_or_before_date():
    rows = []
    for as_of, talent in (("2025-01-01", 70.0), ("2026-01-01", 80.0)):
        for i in range(11):
            rows.append(
                {
                    "as_of": as_of,
                    "team": "A",
                    "player": f"P{i}",
                    "club": f"Club {i}",
                    "position": "MID",
                    "talent": talent,
                    "talent_se": 2,
                    "exp_minutes": 0.9,
                    "available": True,
                    "age": 25,
                    "is_starter": True,
                }
            )
    lookup = SquadLookup(pd.DataFrame(rows))
    assert lookup.get("A", pd.Timestamp("2025-06-01"))["xi_rating"] == 70.0
    assert lookup.get("A", pd.Timestamp("2026-06-01"))["xi_rating"] == 80.0
    assert np.isnan(lookup.get("A", pd.Timestamp("2024-01-01"))["xi_rating"])
