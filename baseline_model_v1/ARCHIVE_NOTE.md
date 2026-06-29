# baseline_model_v1 — frozen baseline (pre Round-of-32)

This is a complete, runnable snapshot of the **original** World Cup model as it
stood after the group-stage forecasts (data through 2026-06-19, model trained
through 2026-06-15). It was frozen on 2026-06-29, immediately before the
Round-of-32 model overhaul.

It is preserved so the old and new models can be compared against actual results.
This was the model that **systematically under-priced favorites** (e.g. France
85% vs Iraq, Argentina 90% vs Jordan, Spain 78% vs Saudi Arabia).

## What's here

A full copy of the repo at the freeze point: `src/`, `config.toml`,
`pyproject.toml`, `data/`, `artifacts/` (trained `model.joblib` + `metrics.json`
+ `team_state.json`), `outputs/`, `predictions_log/`, `PREDICTIONS.md`,
`scripts/`, `tests/`.

## How to run it (without disturbing the updated model)

From inside this folder:

```bash
cd baseline_model_v1
pip install -e .
worldcup predict --config config.toml
worldcup simulate --config config.toml
```

All paths in this `config.toml` are relative, so it reads and writes only inside
`baseline_model_v1/`. The live/updated model in the repo root is unaffected.

## Headline baseline metrics (holdout)

- Outcome log loss: 0.8620 (Elo baseline 0.8924)
- Ranked probability score: 0.1680 (Elo baseline 0.1763)
- Calibration error: 0.0136 (aggregate — hides the favorite-tail bias)

Do **not** edit files in this folder; it is the reference point for the
before/after comparison.
