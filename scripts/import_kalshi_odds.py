"""Import Kalshi closing prices into data/input/match_features.csv as decimal
odds, so the model's published predictions can be anchored to the market
(see [market] blend_weight in config.toml and WorldCupModel.blend_market).

Kalshi winner prices are quoted in cents 0-100 and read as an implied
probability percentage, so decimal odds = 100 / price. The vig is removed later
by FeatureBuilder._market_probabilities, which renormalises 1/odds across
home/draw/away.

This script only READS the Kalshi snapshot; it does not modify anything under
sports_betting_performance/. It upserts the home_odds/draw_odds/away_odds columns
into match_features.csv, preserving any existing columns (e.g. xG).

Run:  python scripts/import_kalshi_odds.py [path/to/kalshi_odds.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldcup_predictor.data import normalize_team  # noqa: E402

DEFAULT_KALSHI = ROOT / "sports_betting_performance" / "data" / "kalshi_odds.csv"
MATCH_FEATURES = ROOT / "data" / "input" / "match_features.csv"
ODDS_COLUMNS = ["home_odds", "draw_odds", "away_odds"]
KEYS = ["date", "home_team", "away_team"]


def price_to_decimal(price: float) -> float | None:
    """Kalshi cent price (implied probability %) -> decimal odds."""
    value = pd.to_numeric(price, errors="coerce")
    if not pd.notna(value) or value <= 0 or value >= 100:
        return None
    return float(100.0 / value)


def main() -> None:
    kalshi_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_KALSHI
    if not kalshi_path.is_file():
        raise SystemExit(f"Kalshi odds file not found: {kalshi_path}")
    kalshi = pd.read_csv(kalshi_path)

    rows = []
    for row in kalshi.itertuples():
        home_odds = price_to_decimal(getattr(row, "winner_home_price", None))
        draw_odds = price_to_decimal(getattr(row, "winner_draw_price", None))
        away_odds = price_to_decimal(getattr(row, "winner_away_price", None))
        if home_odds is None or draw_odds is None or away_odds is None:
            continue
        rows.append(
            {
                "date": pd.Timestamp(row.date).date().isoformat(),
                "home_team": normalize_team(row.home_team),
                "away_team": normalize_team(row.away_team),
                "home_odds": round(home_odds, 4),
                "draw_odds": round(draw_odds, 4),
                "away_odds": round(away_odds, 4),
            }
        )
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        raise SystemExit("No usable 3-way odds found in the Kalshi file.")

    if MATCH_FEATURES.is_file() and MATCH_FEATURES.stat().st_size:
        existing = pd.read_csv(MATCH_FEATURES)
        existing["date"] = pd.to_datetime(existing["date"]).dt.date.astype(str)
        existing["home_team"] = existing["home_team"].map(normalize_team)
        existing["away_team"] = existing["away_team"].map(normalize_team)
        # Drop any prior odds columns so the merge re-adds fresh values.
        existing = existing.drop(columns=ODDS_COLUMNS, errors="ignore")
        merged = existing.merge(incoming, on=KEYS, how="outer")
    else:
        merged = incoming

    MATCH_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values(KEYS).to_csv(MATCH_FEATURES, index=False)
    print(
        f"wrote {len(incoming)} fixtures' odds into {MATCH_FEATURES} "
        f"({len(merged)} total rows)"
    )


if __name__ == "__main__":
    main()
