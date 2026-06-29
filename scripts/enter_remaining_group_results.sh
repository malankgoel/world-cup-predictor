#!/usr/bin/env bash
# Enter the 40 remaining GROUP-STAGE results for WC 2026, then rebuild the model
# for the Round of 32. Replace __H__ (home goals) and __A__ (away goals) on each
# line with the actual scoreline, then run:
#     bash scripts/enter_remaining_group_results.sh
# (Unedited __H__/__A__ placeholders fail cleanly -- worldcup expects integers.)
#
# DO NOT add any knockout game here (e.g. the R32 Canada vs South Africa played
# today). Leaving every knockout out is what keeps this a true, pre-Round-of-32
# forecast with no knockout leakage.
#
# Optional: append  --home-xg <x> --away-xg <x>  to any line to also feed xG into
# the team-state update.
set -euo pipefail
cd "$(dirname "$0")/.."

# --- 2026-06-20 ---
worldcup update --date 2026-06-20 --home "Ecuador" --away "Curaçao" --home-score __H__ --away-score __A__ --city "Kansas City" --country "United States"
worldcup update --date 2026-06-20 --home "Germany" --away "Ivory Coast" --home-score __H__ --away-score __A__ --city "Toronto" --country "Canada"
worldcup update --date 2026-06-20 --home "Netherlands" --away "Sweden" --home-score __H__ --away-score __A__ --city "Houston" --country "United States"
worldcup update --date 2026-06-20 --home "Tunisia" --away "Japan" --home-score __H__ --away-score __A__ --city "Guadalupe" --country "Mexico"
# --- 2026-06-21 ---
worldcup update --date 2026-06-21 --home "Belgium" --away "Iran" --home-score __H__ --away-score __A__ --city "Inglewood" --country "United States"
worldcup update --date 2026-06-21 --home "New Zealand" --away "Egypt" --home-score __H__ --away-score __A__ --city "Vancouver" --country "Canada"
worldcup update --date 2026-06-21 --home "Spain" --away "Saudi Arabia" --home-score __H__ --away-score __A__ --city "Atlanta" --country "United States"
worldcup update --date 2026-06-21 --home "Uruguay" --away "Cape Verde" --home-score __H__ --away-score __A__ --city "Miami Gardens" --country "United States"
# --- 2026-06-22 ---
worldcup update --date 2026-06-22 --home "Argentina" --away "Austria" --home-score __H__ --away-score __A__ --city "Arlington" --country "United States"
worldcup update --date 2026-06-22 --home "France" --away "Iraq" --home-score __H__ --away-score __A__ --city "Philadelphia" --country "United States"
worldcup update --date 2026-06-22 --home "Jordan" --away "Algeria" --home-score __H__ --away-score __A__ --city "Santa Clara" --country "United States"
worldcup update --date 2026-06-22 --home "Norway" --away "Senegal" --home-score __H__ --away-score __A__ --city "East Rutherford" --country "United States"
# --- 2026-06-23 ---
worldcup update --date 2026-06-23 --home "Colombia" --away "DR Congo" --home-score __H__ --away-score __A__ --city "Zapopan" --country "Mexico"
worldcup update --date 2026-06-23 --home "England" --away "Ghana" --home-score __H__ --away-score __A__ --city "Foxborough" --country "United States"
worldcup update --date 2026-06-23 --home "Panama" --away "Croatia" --home-score __H__ --away-score __A__ --city "Toronto" --country "Canada"
worldcup update --date 2026-06-23 --home "Portugal" --away "Uzbekistan" --home-score __H__ --away-score __A__ --city "Houston" --country "United States"
# --- 2026-06-24 ---
worldcup update --date 2026-06-24 --home "Bosnia and Herzegovina" --away "Qatar" --home-score __H__ --away-score __A__ --city "Seattle" --country "United States"
worldcup update --date 2026-06-24 --home "Canada" --away "Switzerland" --home-score __H__ --away-score __A__ --city "Vancouver" --country "Canada" --no-neutral
worldcup update --date 2026-06-24 --home "Mexico" --away "Czech Republic" --home-score __H__ --away-score __A__ --city "Mexico City" --country "Mexico" --no-neutral
worldcup update --date 2026-06-24 --home "Morocco" --away "Haiti" --home-score __H__ --away-score __A__ --city "Atlanta" --country "United States"
worldcup update --date 2026-06-24 --home "Scotland" --away "Brazil" --home-score __H__ --away-score __A__ --city "Miami Gardens" --country "United States"
worldcup update --date 2026-06-24 --home "South Africa" --away "South Korea" --home-score __H__ --away-score __A__ --city "Guadalupe" --country "Mexico"
# --- 2026-06-25 ---
worldcup update --date 2026-06-25 --home "Curaçao" --away "Ivory Coast" --home-score __H__ --away-score __A__ --city "Philadelphia" --country "United States"
worldcup update --date 2026-06-25 --home "Ecuador" --away "Germany" --home-score __H__ --away-score __A__ --city "East Rutherford" --country "United States"
worldcup update --date 2026-06-25 --home "Japan" --away "Sweden" --home-score __H__ --away-score __A__ --city "Arlington" --country "United States"
worldcup update --date 2026-06-25 --home "Paraguay" --away "Australia" --home-score __H__ --away-score __A__ --city "Santa Clara" --country "United States"
worldcup update --date 2026-06-25 --home "Tunisia" --away "Netherlands" --home-score __H__ --away-score __A__ --city "Kansas City" --country "United States"
worldcup update --date 2026-06-25 --home "United States" --away "Turkey" --home-score __H__ --away-score __A__ --city "Inglewood" --country "United States" --no-neutral
# --- 2026-06-26 ---
worldcup update --date 2026-06-26 --home "Cape Verde" --away "Saudi Arabia" --home-score __H__ --away-score __A__ --city "Houston" --country "United States"
worldcup update --date 2026-06-26 --home "Egypt" --away "Iran" --home-score __H__ --away-score __A__ --city "Seattle" --country "United States"
worldcup update --date 2026-06-26 --home "New Zealand" --away "Belgium" --home-score __H__ --away-score __A__ --city "Vancouver" --country "Canada"
worldcup update --date 2026-06-26 --home "Norway" --away "France" --home-score __H__ --away-score __A__ --city "Foxborough" --country "United States"
worldcup update --date 2026-06-26 --home "Senegal" --away "Iraq" --home-score __H__ --away-score __A__ --city "Toronto" --country "Canada"
worldcup update --date 2026-06-26 --home "Uruguay" --away "Spain" --home-score __H__ --away-score __A__ --city "Zapopan" --country "Mexico"
# --- 2026-06-27 ---
worldcup update --date 2026-06-27 --home "Algeria" --away "Austria" --home-score __H__ --away-score __A__ --city "Kansas City" --country "United States"
worldcup update --date 2026-06-27 --home "Colombia" --away "Portugal" --home-score __H__ --away-score __A__ --city "Miami Gardens" --country "United States"
worldcup update --date 2026-06-27 --home "Croatia" --away "Ghana" --home-score __H__ --away-score __A__ --city "Philadelphia" --country "United States"
worldcup update --date 2026-06-27 --home "DR Congo" --away "Uzbekistan" --home-score __H__ --away-score __A__ --city "Atlanta" --country "United States"
worldcup update --date 2026-06-27 --home "Jordan" --away "Argentina" --home-score __H__ --away-score __A__ --city "Arlington" --country "United States"
worldcup update --date 2026-06-27 --home "Panama" --away "England" --home-score __H__ --away-score __A__ --city "East Rutherford" --country "United States"

# Rebuild the goal model + team states on the full, completed group stage:
worldcup --config config.toml train

# (Optional) import current Round-of-32 market odds, if you have added them to
# sports_betting_performance/data/kalshi_odds.csv, so published predictions get
# the market anchor:
# python scripts/import_kalshi_odds.py

# Forecast the bracket from the actual group standings:
worldcup simulate --config config.toml
