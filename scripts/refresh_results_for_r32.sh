#!/usr/bin/env bash
# Pull the REAL international results (martj42 dataset, the same source the model
# was already using), keep only games through the end of the group stage, and
# rebuild the model so it is a true, pre-Round-of-32 forecast.
#
# Why trim: a plain results refresh now also contains the knockout games that
# have already been played (e.g. the R32 Canada vs South Africa game today).
# With include_tournament = true the simulator would pin those as observed and
# the bracket would no longer be a forecast. We therefore drop every
# "FIFA World Cup" match on or after the first knockout date (read from the
# schedule, currently 2026-06-28) before training.
#
# Run from the repo root:  bash scripts/refresh_results_for_r32.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">> 1/6  refreshing real results + shootouts from martj42 (schedule untouched)"
# Shootouts matter from the Round of 16 on: a knockout level after extra time is
# decided on penalties, and the model reads the advancing team from shootouts.csv
# (merged into the results' winner column by cli.merge_shootout_winners).
python -c "from worldcup_predictor.data import download_results, download_shootouts; download_results('data/raw/results.csv'); download_shootouts('data/raw/shootouts.csv')"

echo ">> 2/6  trimming knockout games so the forecast stays pre-Round-of-32"
python - <<'PY'
import pandas as pd
sched = pd.read_csv("data/input/schedule_2026.csv")
sched["date"] = pd.to_datetime(sched["date"])
cutoff = sched.loc[sched["stage"] != "group", "date"].min()   # first knockout date

res = pd.read_csv("data/raw/results.csv")
res["date"] = pd.to_datetime(res["date"])
is_wc_ko = (res["tournament"] == "FIFA World Cup") & (res["date"] >= cutoff)
dropped = int(is_wc_ko.sum())
kept = res[~is_wc_ko].copy()
kept["date"] = kept["date"].dt.date
kept.sort_values("date").to_csv("data/raw/results.csv", index=False)

wc = kept[(kept["tournament"] == "FIFA World Cup") & (pd.to_datetime(kept["date"]) >= "2026-06-11")]
print(f"   first knockout date (cutoff): {cutoff.date()}")
print(f"   dropped {dropped} World Cup knockout match(es) dated >= {cutoff.date()}")
print(f"   kept {len(wc)} / 72 World Cup group matches")
if len(wc) < 72:
    print(f"   NOTE: {72 - len(wc)} group games not yet in the feed; rerun once they post.")
PY

echo ">> 3/6  importing current market odds (if any) for the market blend"
# Reads sports_betting_performance/data/kalshi_odds.csv and writes the implied
# odds into match_features.csv. Add the Round-of-32 fixtures' odds to that file
# to anchor the R32 lines to the market; until then only group odds are present
# and this is a harmless no-op for R32.
python scripts/import_kalshi_odds.py \
  || echo "   (no new odds to import — R32 lines will be pure-model for now)"

echo ">> 4/6  retraining the goal model + team states on the completed group stage"
worldcup --config config.toml train

echo ">> 5/6  simulating the bracket from the actual group standings"
worldcup --config config.toml simulate

echo ">> 6/6  predicting every Round-of-32 matchup (1X2 + who-advances moneyline)"
# Only works once all 72 group games are in the feed; otherwise it explains what
# is still missing without failing the whole run.
worldcup --config config.toml predict-knockouts \
  || echo "   (R32 matchups not fully decided yet — rerun once all group games post)"

echo ">> done. Title/advancement odds in PREDICTIONS.md; per-match R32 lines"
echo "   (90-min 1X2 + home_advance/away_advance) in outputs/knockout_predictions.csv"
