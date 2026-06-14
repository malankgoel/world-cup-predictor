import pandas as pd

from worldcup_predictor import report


def _config(tmp_path):
    (tmp_path / "outputs").mkdir()
    return {
        "_root": str(tmp_path),
        "paths": {
            "report": "PREDICTIONS.md",
            "predictions": "outputs/match_predictions.csv",
            "simulation": "outputs/tournament_probabilities.csv",
            "metrics": "artifacts/metrics.json",
            "predictions_log": "predictions_log",
        },
    }


def test_write_report_renders_both_sections(tmp_path):
    config = _config(tmp_path)
    pd.DataFrame(
        {
            "date": ["2026-06-11"],
            "group": ["A"],
            "home_team": ["Mexico"],
            "away_team": ["South Africa"],
            "home_win": [0.74],
            "draw": [0.18],
            "away_win": [0.08],
            "most_likely_score": ["2 - 0"],
            "expected_home_goals": [2.1],
            "expected_away_goals": [0.5],
            "over_2_5": [0.49],
            "both_teams_to_score": [0.36],
        }
    ).to_csv(tmp_path / "outputs" / "match_predictions.csv", index=False)
    pd.DataFrame(
        {
            "team": ["Spain", "Argentina"],
            "champion": [0.19, 0.16],
            "reach_final": [0.30, 0.28],
            "reach_semi_final": [0.40, 0.39],
        }
    ).to_csv(tmp_path / "outputs" / "tournament_probabilities.csv", index=False)

    path = report.write_report(config, results_through="2026-06-10")
    text = path.read_text()

    assert path.name == "PREDICTIONS.md"
    assert "# World Cup 2026 — Model Predictions" in text
    assert "data through 2026-06-10" in text
    assert "Title & advancement probabilities" in text
    # Champion column is the lead sort: Spain (19%) before Argentina (16%).
    assert text.index("Spain") < text.index("Argentina")
    assert "19.0%" in text
    assert "Group-stage match predictions" in text
    assert "Mexico vs South Africa" in text
    assert "74% / 18% / 8%" in text


def test_write_report_handles_no_outputs(tmp_path):
    config = _config(tmp_path)
    path = report.write_report(config)
    assert "No predictions yet" in path.read_text()
