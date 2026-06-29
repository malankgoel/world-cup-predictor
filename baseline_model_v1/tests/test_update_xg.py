from argparse import Namespace

import pandas as pd

from worldcup_predictor.cli import record_match_xg


def _args(**overrides):
    base = dict(
        date="2026-06-13",
        home="Brazil",
        away="Morocco",
        home_xg=2.4,
        away_xg=0.8,
    )
    base.update(overrides)
    return Namespace(**base)


def test_record_match_xg_appends_new_row(tmp_path):
    mf = tmp_path / "match_features.csv"
    config = {"paths": {"match_features": str(mf)}}

    assert record_match_xg(config, _args()) is True

    frame = pd.read_csv(mf)
    row = frame[(frame.home_team == "Brazil") & (frame.away_team == "Morocco")]
    assert len(row) == 1
    assert float(row.iloc[0].home_xg) == 2.4
    assert float(row.iloc[0].away_xg) == 0.8


def test_record_match_xg_updates_existing_row(tmp_path):
    mf = tmp_path / "match_features.csv"
    pd.DataFrame(
        [{"date": "2026-06-13", "home_team": "Brazil", "away_team": "Morocco",
          "home_xg": 1.0, "away_xg": 1.0}]
    ).to_csv(mf, index=False)
    config = {"paths": {"match_features": str(mf)}}

    record_match_xg(config, _args(home_xg=2.4, away_xg=0.8))

    frame = pd.read_csv(mf)
    assert len(frame) == 1  # upsert, not duplicate
    assert float(frame.iloc[0].home_xg) == 2.4


def test_record_match_xg_is_noop_without_xg(tmp_path):
    mf = tmp_path / "match_features.csv"
    config = {"paths": {"match_features": str(mf)}}
    assert record_match_xg(config, _args(home_xg=None, away_xg=None)) is False
    assert not mf.exists()
