"""Append-only forecast logbook.

Every time predictions or a tournament simulation are produced, a snapshot is
archived under ``predictions_log/`` so the evolution of the forecast across the
tournament is preserved. Snapshots are keyed by the latest played-result date
(the "data through" point), so each World Cup matchday that you fold in with
``worldcup update`` produces a new, comparable checkpoint and the earlier ones
are never overwritten.

Layout::

    predictions_log/
        index.jsonl                  # one JSON line per recorded run
        <results_through>/           # one folder per data-through checkpoint
            match_predictions.csv
            tournament_probabilities.csv
            manifest.json

Logging is best-effort: a failure here never breaks predict/simulate.
"""
from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def latest_result_date(results: pd.DataFrame) -> str | None:
    """ISO date of the most recent played match, or None when there are none."""
    if results.empty or "date" not in results:
        return None
    return pd.to_datetime(results["date"]).max().date().isoformat()


def _top_champions(simulation: pd.DataFrame, count: int = 5) -> list[dict]:
    if "champion" not in simulation or "team" not in simulation:
        return []
    ordered = simulation.sort_values("champion", ascending=False).head(count)
    return [
        {"team": str(row["team"]), "champion": round(float(row["champion"]), 4)}
        for _, row in ordered.iterrows()
    ]


def record(
    config: dict,
    results: pd.DataFrame,
    kind: str,
    output: pd.DataFrame,
) -> Path | None:
    """Snapshot one ``predict`` or ``simulate`` output into the logbook.

    Parameters
    ----------
    kind:
        ``"predict"`` or ``"simulate"``.
    output:
        The data frame that was just written to the live ``outputs/`` path.
    """
    try:
        root = Path(config.get("_root", "."))
        log_dir = root / config["paths"].get("predictions_log", "predictions_log")
        through = latest_result_date(results) or "pretournament"
        snapshot_dir = log_dir / through
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        live_path = Path(config["paths"][_LIVE_PATH_KEY[kind]])
        filename = live_path.name
        output.to_csv(snapshot_dir / filename, index=False)

        recorded_at = datetime.now(UTC).isoformat(timespec="seconds")
        model_through = None
        metrics_path = root / config["paths"].get("metrics", "")
        if metrics_path.is_file():
            try:
                model_through = json.loads(metrics_path.read_text()).get(
                    "training_through"
                )
            except (json.JSONDecodeError, OSError):
                model_through = None

        manifest_path = snapshot_dir / "manifest.json"
        manifest: dict = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                manifest = {}
        manifest.update(
            {
                "results_through": through,
                "played_matches": int(len(results)),
                "model_training_through": model_through,
                "git_commit": _git_commit(root),
                f"{kind}_recorded_at": recorded_at,
                f"{kind}_file": filename,
            }
        )
        if kind == "simulate":
            manifest["top_champions"] = _top_champions(output)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

        entry = {
            "recorded_at": recorded_at,
            "kind": kind,
            "results_through": through,
            "played_matches": int(len(results)),
            "snapshot": str(snapshot_dir.relative_to(root)),
        }
        if kind == "simulate":
            entry["top_champions"] = _top_champions(output, count=3)
        with (log_dir / "index.jsonl").open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        return snapshot_dir
    except Exception:  # noqa: BLE001 - logging must never break the main command
        return None


def history(config: dict) -> list[dict]:
    """Return every recorded logbook entry, oldest first."""
    root = Path(config.get("_root", "."))
    index = root / config["paths"].get("predictions_log", "predictions_log") / (
        "index.jsonl"
    )
    if not index.is_file():
        return []
    entries = []
    for line in index.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


_LIVE_PATH_KEY = {"predict": "predictions", "simulate": "simulation"}
