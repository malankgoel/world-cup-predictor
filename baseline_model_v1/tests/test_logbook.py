import json

import pandas as pd

from worldcup_predictor import logbook


def _config(tmp_path):
    return {
        "_root": str(tmp_path),
        "paths": {
            "predictions_log": "predictions_log",
            "predictions": "outputs/match_predictions.csv",
            "simulation": "outputs/tournament_probabilities.csv",
            "metrics": "artifacts/metrics.json",
        },
    }


def test_latest_result_date_picks_most_recent():
    results = pd.DataFrame({"date": pd.to_datetime(["2026-06-01", "2026-06-10"])})
    assert logbook.latest_result_date(results) == "2026-06-10"


def test_latest_result_date_handles_empty():
    assert logbook.latest_result_date(pd.DataFrame()) is None


def test_record_snapshots_keyed_by_data_through(tmp_path):
    config = _config(tmp_path)
    results = pd.DataFrame({"date": pd.to_datetime(["2026-06-01", "2026-06-10"])})
    predict = pd.DataFrame({"home_team": ["Mexico"], "home_win": [0.74]})
    simulate = pd.DataFrame({"team": ["Spain"], "champion": [0.19]})

    snapshot = logbook.record(config, results, "predict", predict)
    logbook.record(config, results, "simulate", simulate)

    assert snapshot is not None
    assert snapshot.name == "2026-06-10"
    assert (snapshot / "match_predictions.csv").is_file()
    assert (snapshot / "tournament_probabilities.csv").is_file()

    manifest = json.loads((snapshot / "manifest.json").read_text())
    assert manifest["results_through"] == "2026-06-10"
    assert manifest["played_matches"] == 2
    assert manifest["top_champions"][0]["team"] == "Spain"

    history = logbook.history(config)
    assert [entry["kind"] for entry in history] == ["predict", "simulate"]


def test_record_stamps_local_run_time(tmp_path):
    config = _config(tmp_path)
    results = pd.DataFrame({"date": pd.to_datetime(["2026-06-10"])})
    predict = pd.DataFrame({"home_team": ["Mexico"], "home_win": [0.74]})

    snapshot = logbook.record(config, results, "predict", predict)

    manifest = json.loads((snapshot / "manifest.json").read_text())
    # Human-readable local date + time of day, e.g. "2026-06-14 13:48:27 EDT".
    assert "last_run_at" in manifest
    assert manifest["last_run_at"] == manifest["predict_last_run_at"]
    stamp = manifest["last_run_at"]
    assert stamp[:10] == manifest["predict_recorded_at"][:10]  # same date
    assert ":" in stamp.split(" ")[1]  # includes a HH:MM:SS time

    entry = logbook.history(config)[-1]
    assert entry["last_run_at"] == stamp


def test_record_is_best_effort_on_bad_config(tmp_path):
    # Missing the path key for the kind: must not raise, just return None.
    config = {"_root": str(tmp_path), "paths": {"predictions_log": "log"}}
    results = pd.DataFrame({"date": pd.to_datetime(["2026-06-10"])})
    assert logbook.record(config, results, "predict", pd.DataFrame()) is None
