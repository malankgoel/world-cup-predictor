"""Expanding-window backtest: train before each cutoff, test on the next window.

A single chronological holdout (what `train` reports) can be noisy. This walks
several fold boundaries instead: for fold *i* the model trains on every match
before ``cutoffs[i]`` and is evaluated on matches in
``[cutoffs[i], cutoffs[i + 1])`` (the final fold tests on everything on or after
the last cutoff). It reuses the same walk-forward feature frame produced by
``FeatureBuilder.training_frame``, so a test fold's features never include its
own or any later result.
"""
from __future__ import annotations

import pandas as pd

from .model import WorldCupModel


def expanding_window_backtest(
    features: pd.DataFrame,
    *,
    cutoffs: list,
    model_parameters: dict | None = None,
    model_adjustments: dict | None = None,
    calibration_temperature: float | None = None,
    half_life_years: float = 4.0,
    min_train: int = 500,
) -> pd.DataFrame:
    """Return one row of holdout metrics per fold.

    ``features`` must be a walk-forward feature frame (the output of
    ``FeatureBuilder.training_frame``). ``cutoffs`` is a list of fold-boundary
    dates in ascending order.
    """
    features = features.sort_values("date").reset_index(drop=True)
    bounds = sorted(pd.Timestamp(cutoff) for cutoff in cutoffs)
    rows = []
    for index, start in enumerate(bounds):
        end = bounds[index + 1] if index + 1 < len(bounds) else None
        train = features[features["date"] < start]
        if end is None:
            test = features[features["date"] >= start]
        else:
            test = features[(features["date"] >= start) & (features["date"] < end)]
        if len(train) < min_train or test.empty:
            continue
        model = WorldCupModel(
            model_parameters,
            adjustments=model_adjustments,
            calibration_temperature=calibration_temperature,
        )
        model.fit_window(train, half_life_years=half_life_years)
        metrics = model.evaluate(test)
        metrics["fold_start"] = start.date().isoformat()
        metrics["train_matches"] = int(len(train))
        metrics["test_matches"] = int(len(test))
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_backtest(result: pd.DataFrame) -> dict[str, float]:
    """Mean of every numeric metric column across folds."""
    if result.empty:
        return {}
    numeric = result.select_dtypes("number").drop(
        columns=["train_matches", "test_matches"], errors="ignore"
    )
    return {column: float(value) for column, value in numeric.mean().items()}
