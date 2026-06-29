# Round-of-32 model overhaul — what shipped, what the backtest rejected

This documents the R32 changes to the **updated (v2)** model in the repo root.
The frozen pre-overhaul model lives in `baseline_model_v1/` for comparison.

The headline complaint was that the model **severely under-priced favorites**.
The first thing the overhaul did was quantify it on the 2022 World Cup
out-of-sample backtest (train only on pre-2022 internationals, score the 64
actual matches). Favorites in the 0.55–0.80 predicted band won **+18pp more
often than predicted** in 2022 and **+5pp** in 2018 — real, and persistent.

The important and slightly counter-intuitive finding: **most ways of "making the
model more confident in favorites" do not survive out-of-sample testing.** 2022
was an upset-heavy tournament (Saudi over Argentina, Japan over Germany & Spain,
Morocco to the semis), and log loss punishes confident-and-wrong hard. So blanket
sharpening helps 2018 but actively hurts 2022. The fix that *does* generalize is
anchoring each match to the **betting market**, which is well-calibrated and
adapts per matchup instead of betting on a tournament being chalky or chaotic.

## Shipped

**1. Market anchor (the real favorite fix).** A log-opinion-pool blend of the
model's 1X2 with the vig-removed market 1X2:
`p_blend ∝ p_model^(1−w) · p_market^w`, `w = [market].blend_weight = 0.5`.
Applied to published predictions whenever odds are present
(`WorldCupModel.blend_market`, wired into `predict_schedule`). `scripts/
import_kalshi_odds.py` converts Kalshi prices to decimal odds in
`data/input/match_features.csv` (it only **reads** the betting folder). Effect on
the live group fixtures — favorites that were under-priced get lifted toward the
market, and the few that were over-priced get moderated:

| Fixture | model | blended | market |
|---|---|---|---|
| Spain v Cape Verde | 0.75 | 0.85 | 0.91 |
| Portugal v DR Congo | 0.56 | 0.68 | 0.77 |
| Brazil v Morocco | 0.42 | 0.50 | 0.58 |
| France v Senegal | 0.54 | 0.60 | 0.66 |
| Canada v Qatar (over-priced) | 0.86 | 0.82 | 0.77 |

**2. Recency half-life 4.0 → 3.0 years** (`[training].recency_half_life_years`).
The one *training* change that improved holdout log loss on **both** tournaments
(2022 1.1033→1.0981, 2018 0.9735→0.9675). Current squad strength matters more
than five-year-old form.

**3. Monotonic constraints on the goal model** (`MONOTONIC_SIGNS` in `model.py`).
A `HistGradientBoosting` model predicts leaf averages and can't extrapolate past
the strongest mismatches it trained on, so it compresses elite-vs-minnow rates.
Pinning the sign of each unambiguous strength signal (own strength +, opponent
strength −) stops perverse splits. Backtest-neutral on its own, but it is also
what makes the market-probability feature behave monotonically if it ever enters
the trained model, so it's kept as a principled safeguard.

**4. Tournament-importance weight power 0.5 → 0.75**
(`[training].importance_weight_power`). Up-weights real tournament football over
friendlies. Backtest-neutral (helps 2022, marginally hurts 2018, both inside
64-match noise); kept because it's principled, not because it's a proven win.

**5. Per-match noise made configurable** (`[simulation].match_noise_sigma_scale`,
plumbed through `TournamentSimulator`). See "rejected" below for why the default
is **1.0**, not 0.0.

All 45 unit tests pass (4 new, covering the blend, the monotonic attachment, and
the importance weighting); `ruff` is clean. The live model was retrained and the
published match predictions were regenerated with the blend on display.

## Tested and rejected (kept the original behaviour)

**Global temperature sharpening.** Sweeping the 1X2 temperature on both
tournaments: average log loss is *minimised at temperature 1.0* and gets worse as
you sharpen, because 2022 punishes confidence. This is why the original author
left it unpinned — confirmed, not changed.

**Cutting the "double-counted" per-match simulation noise (proposed item #2).**
In theory the per-match rate shock duplicates the variance already in the score
draw. But against the **actual** 2022 bracket, setting it to 0.0 made advancement
calibration **worse at every stage** (total Brier 0.452 → 0.464), because
`team_strength_scale` was tuned *with* that shock present and 2022 rewards more
spread. The knob is exposed so it can be re-tuned jointly with
`team_strength_scale`, but the default stays at the empirically-better 1.0.

## Still outstanding — needs the actual results (item #7)

`data/raw/results.csv` has the 2026 fixtures with **NA scores**, and
`load_results` drops NA rows — so the model currently treats the whole tournament
as unplayed. To make the bracket and team-states reflect what really happened in
the group stage (the biggest single lever for tightening R32 specifically), the
real group scores must be entered, e.g. `worldcup update --date … --home … --away
… --home-score … --away-score …` (with `--home-xg/--away-xg` when available),
then `worldcup train`. Once R32 fixtures and their Kalshi odds are added, the
market anchor applies to them automatically.
