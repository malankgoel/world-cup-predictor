# Tightening the model for the Round of 32 and beyond

_Why the model keeps under-pricing favorites, and the concrete levers to fix it. Based on a read of `src/worldcup_predictor/` (model.py, tournament.py, features.py) and `config.toml`._

## The core problem in one sentence

Every layer of the pipeline adds a little mean-zero noise or a little shrinkage to the goal rates. Each is individually defensible, but **adding symmetric noise to a goal-rate gap is not symmetric on win probability — it bleeds the favorite toward 50%.** Stack three or four of these layers and a true 90% favorite comes out at 80%. That is exactly the pattern you saw (France 85% vs Iraq, Argentina 90% vs Jordan, Spain 78% vs Saudi — all visibly soft vs the market).

There is no single bug; it's accumulated regression-to-the-mean. Below, ranked by expected impact.

---

## 1. Anchor to the betting market — biggest, easiest win

The model *already declares* `home_market_probability`, `draw_market_probability`, `away_market_probability` as features (see `model_features` in `artifacts/metrics.json`), but they are **completely empty** — `data/input/match_features.csv` has zero non-null market cells, so they never become active features. The single best predictor of a favorite's true win rate is the closing line, and it's currently unused.

Two ways to use it, in order of preference:

- **Blend at the output (log-opinion pool).** Take the model's 1X2 vector and the market's implied (vig-removed) vector and combine in log space: `p ∝ p_model^(1−w) · p_market^w`, with `w ≈ 0.4–0.6`. This is robust, hard to overfit, and directly cures tail compression because the market does not under-price favorites.
- **Feed it as a feature** (populate the existing columns). Weaker, because the GBM will still partially shrink it.

You have Kalshi odds being collected in `sports_betting_performance/data/kalshi_odds.csv`. Per your instruction I'm leaving that folder alone, but the *main* model can read a copy of those closing probabilities. This alone probably closes most of the gap.

## 2. Stop stacking noise layers in the simulation

In `tournament.py._play()` each simulated match multiplies both goal rates by an independent lognormal shock (`sigma` up to 0.35), **on top of** the per-tournament `_apply_team_shock` (`TEAM_STRENGTH_SCALE = 0.20`). Two independent variance injections, both of which widen the rate gap distribution and therefore pull favorites' *advancement* probabilities toward the field.

- The per-tournament team shock is the one with a principled justification (it restores between-match correlation). Keep it.
- The per-match `sigma` shock is mostly double-counting — the score is *already* sampled from a Poisson/NB matrix, which is the match-level randomness. Cut it hard (try `sigma → 0`) or fold its intended uncertainty into the team shock, then re-fit `TEAM_STRENGTH_SCALE` against advancement Brier.
- Now that group games are in (`include_tournament = true`), posterior uncertainty is lower, so the shock scale should come *down* for R32 onward regardless.

## 3. The GBM compresses extremes — give it a strength term that extrapolates

`HistGradientBoostingRegressor` (max_leaf_nodes=15, lr=0.03) predicts goal rates from leaf averages, so it **cannot extrapolate past the strongest matchups it saw in training** and systematically pulls elite-vs-minnow rates toward the middle of the training distribution — which is dominated by friendlies and qualifiers, not World Cup mismatches.

- **Ensemble the GBM rate with a log-linear strength rate** (Elo/Bradley–Terry implied, or a Dixon–Coles attack/defense model). The linear model extrapolates; blend the two log-rates. This is the structural fix for tail compression.
- Or add **monotonic constraints** on `elo_diff`, `rank_points_diff`, `xi_rating_diff`, `squad_attack/defense_diff` so strength always moves the rate the right direction with no saturation reversals. `HistGradientBoostingRegressor` supports `monotonic_cst` directly.

## 4. Re-weight training toward tournament football

Favorites convert at a *higher* rate in World Cups than in friendlies (more at stake, fewer experimental lineups). Your training is ~25k matches, mostly low-stakes. `_sample_weights` already multiplies by `sqrt(importance)` — push that harder:

- Raise the importance exponent (e.g. `importance^0.75` or linear) so WC/continental-cup matches dominate the fit.
- Shorten `recency_half_life_years` modestly (4.0 → 3.0) so current squad strength matters more than 5-year-old form.
- This teaches the model the *real* favorite-conversion rate instead of the friendly-diluted one.

## 5. Replace global temperature with tail-aware calibration

`calibration_temperature ≈ 0.97` is fit on a ~5,000-match validation holdout (mostly non-tournament) and is a **single global scalar** — it cannot fix miscalibration that lives specifically in the high-confidence tail. A favorite-undervaluation problem is a tail problem, so global temperature is the wrong tool.

- Fit **isotonic regression or a logistic recalibration on the favorite's win probability** (binned), using historical tournament matches. This can sharpen the 70–95% range without touching coin-flips.
- At minimum, evaluate calibration *conditioned on favorite strength buckets* (0.6–0.7, 0.7–0.8, 0.8–0.9, 0.9+) rather than the single aggregate `calibration_error = 0.0136`, which hides tail bias.

## 6. Tame the draw share

In low-scoring independent-Poisson matrices, draw probability is structurally inflated, and every point of excess draw comes out of the favorite's win column. Your fitted Dixon–Coles `rho ≈ −0.047` is small.

- Refit `rho` on **tournament matches only**, and check the realized draw rate vs predicted in the validation set. If predicted draws > actual, that's favorite probability leaking into the draw bucket.
- Consider a **bivariate Poisson** (shared component) instead of Dixon–Coles low-score correction — it handles the draw/correlation structure more cleanly.

## 7. R32-specific: actually ingest the group-stage results

You've flipped `include_tournament = true`, but confirm the group results are (a) loaded with real scores — right now `data/raw/results.csv` has `NA` for every 2026 match — and (b) updating each team's attack/defense state via the xG-aware path. For knockouts:

- Backfill the 72 group-stage scorelines (and xG if available) into `results.csv`, then `worldcup update`/retrain so team states reflect *this tournament's* form, not just pre-tournament priors.
- With real evidence in hand, **lower `TEAM_STRENGTH_SCALE`** (less prior uncertainty) — this is the right moment for the model to get more confident in the teams that looked strong.
- Knockouts have no draws (ET → pens). You already branch on `knockout` for extra-time; double-check the ET goal rate (`rate/3`) and the shootout `PENALTY_SKILL_WEIGHT = 0.89` shrink aren't themselves flattening the stronger side.

---

## Suggested order of attack

1. **Market blend (#1)** — do this first; largest effect for least code.
2. **Cut the double noise (#2)** and **re-fit the shock scale** with group results in.
3. **Monotonic constraints / strength ensemble (#3)** — structural, medium effort.
4. **Tournament re-weighting (#4)** and **tail calibration (#5)** — re-train and re-check bucketed calibration.
5. Draw/bivariate-Poisson (#6) and the R32 data hygiene (#7) as cleanup.

After each change, re-run the 2022 out-of-sample backtest (`scripts/backtest_2022.py`) and check **calibration in the 0.7–0.95 favorite bucket specifically**, not just aggregate log loss — the aggregate is what's been hiding this.
