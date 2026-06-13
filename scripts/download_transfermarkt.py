"""Fetch Transfermarkt market values for every player in data/input/squads.csv.

Market value covers virtually every professional worldwide, including the
Gulf, African, and Liga MX players that EA SPORTS FC does not license. It is
therefore the signal we use to replace the flat-floor imputation that EA
misses fall back to (see scripts/apply_transfermarkt.py).

Transfermarkt blocks shared/cloud IPs, so the public demo instance is unusable
for a bulk pull. Self-host the felipeall/transfermarkt-api scraper first:

    git clone https://github.com/felipeall/transfermarkt-api.git
    cd transfermarkt-api && docker compose up -d        # serves :8000

Then, from this project:

    python scripts/download_transfermarkt.py             # base url defaults to
                                                         # $TRANSFERMARKT_API or
                                                         # http://localhost:8000

Inputs:
  data/input/squads.csv          built by scripts/build_squads.py
Outputs:
  data/raw/transfermarkt.json    one record per squad player (resumable cache)

The cache is keyed by (team, player); re-running skips players already fetched,
so an interrupted or rate-limited run can simply be restarted.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SQUADS = ROOT / "data" / "input" / "squads.csv"
CACHE = ROOT / "data" / "raw" / "transfermarkt.json"

# Pin values to the pre-tournament snapshot so this stays consistent with the
# squad as_of and the model's cutoff guard (no post-kickoff hype leaks in).
AS_OF = date(2026, 6, 1)
REQUEST_PAUSE = 2.5  # seconds between calls; the scraper is polite by default


def fold(value: str) -> str:
    """Accent-fold and strip punctuation for name comparison."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("-", " ").replace("'", "").replace(".", "")
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_value(raw) -> float:
    """Convert a Transfermarkt money string ("€80.00m", "€500k") to euros."""
    if raw is None:
        return float("nan")
    text = str(raw).strip().lower().replace("€", "").replace(",", "").replace(" ", "")
    match = re.match(r"^([\d.]+)\s*(bn|m|k)?$", text)
    if not match:
        return float("nan")
    number = float(match.group(1))
    scale = {"bn": 1e9, "m": 1e6, "k": 1e3, None: 1.0}[match.group(2)]
    return number * scale


class Client:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _get(self, path: str, **params) -> dict | None:
        url = f"{self.base_url}{path}"
        for attempt in range(4):
            try:
                response = self.session.get(url, params=params, timeout=40)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.json()
            except Exception:
                if attempt == 3:
                    return None
                time.sleep(REQUEST_PAUSE * (attempt + 1))
        return None

    def search(self, name: str) -> list[dict]:
        payload = self._get(f"/players/search/{requests.utils.quote(name)}")
        return (payload or {}).get("results", []) if payload else []

    def market_value(self, player_id: str) -> dict | None:
        return self._get(f"/players/{player_id}/market_value")

    def profile(self, player_id: str) -> dict | None:
        return self._get(f"/players/{player_id}/profile")


def _candidate_age(candidate: dict) -> float:
    age = candidate.get("age")
    try:
        return float(age)
    except (TypeError, ValueError):
        return float("nan")


def _club_name(candidate: dict) -> str:
    club = candidate.get("club")
    if isinstance(club, dict):
        return str(club.get("name", ""))
    return str(club or "")


def choose(name: str, club: str, age: float, candidates: list[dict]) -> tuple[dict | None, float]:
    """Pick the best search hit, scoring on name, club, and age agreement."""
    folded = fold(name)
    name_tokens = set(folded.split())
    best, best_score = None, 0.0
    for candidate in candidates:
        cand_name = fold(candidate.get("name", ""))
        name_ratio = SequenceMatcher(None, folded, cand_name).ratio()
        token_overlap = (
            len(name_tokens & set(cand_name.split())) / max(1, len(name_tokens))
        )
        score = 0.6 * name_ratio + 0.4 * token_overlap
        club_ratio = SequenceMatcher(None, fold(club), fold(_club_name(candidate))).ratio()
        score += 0.25 * club_ratio
        cand_age = _candidate_age(candidate)
        if pd.notna(age) and pd.notna(cand_age):
            score += 0.20 if abs(cand_age - age) <= 1.5 else -0.30
        if score > best_score:
            best, best_score = candidate, score
    return best, best_score


def value_as_of(history: list[dict], current: float) -> tuple[float, str]:
    """Latest historical value dated on/before AS_OF; else fall back to current."""
    dated = []
    for entry in history or []:
        when = pd.to_datetime(entry.get("date"), errors="coerce")
        value = parse_value(entry.get("marketValue"))
        if pd.notna(when) and pd.notna(value):
            dated.append((when.date(), value))
    eligible = [(when, value) for when, value in dated if when <= AS_OF]
    if eligible:
        when, value = max(eligible, key=lambda item: item[0])
        return value, when.isoformat()
    return current, "current"


def load_cache() -> dict[tuple[str, str], dict]:
    if CACHE.exists() and CACHE.stat().st_size:
        records = json.loads(CACHE.read_text())
        return {(record["team"], record["player"]): record for record in records}
    return {}


def save_cache(cache: dict[tuple[str, str], dict]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(list(cache.values()), indent=1))


def main() -> None:
    base_url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("TRANSFERMARKT_API", "http://localhost:8000")
    )
    squads = pd.read_csv(SQUADS)
    client = Client(base_url)
    cache = load_cache()
    pending = [
        row
        for _, row in squads.iterrows()
        if (row["team"], row["player"]) not in cache
    ]
    print(f"{len(squads)} players; {len(pending)} to fetch from {base_url}")

    for index, row in enumerate(pending, 1):
        team, player = row["team"], row["player"]
        club = str(row.get("club", ""))
        age = pd.to_numeric(row.get("age"), errors="coerce")
        record = {
            "team": team,
            "player": player,
            "squad_club": club,
            "squad_age": float(age) if pd.notna(age) else None,
            "tm_id": None,
            "tm_name": None,
            "match_score": None,
            "value_eur_asof": None,
            "value_eur_current": None,
            "asof_date_used": None,
        }
        results = client.search(player)
        time.sleep(REQUEST_PAUSE)
        candidate, score = choose(player, club, age, results)
        if candidate is not None and score >= 0.6:
            player_id = str(candidate.get("id"))
            mv = client.market_value(player_id)
            time.sleep(REQUEST_PAUSE)
            current = parse_value((mv or {}).get("marketValue"))
            asof, used = value_as_of((mv or {}).get("marketValueHistory"), current)
            record.update(
                tm_id=player_id,
                tm_name=candidate.get("name"),
                match_score=round(score, 3),
                value_eur_current=None if pd.isna(current) else current,
                value_eur_asof=None if pd.isna(asof) else asof,
                asof_date_used=used,
            )
        cache[(team, player)] = record
        if index % 25 == 0:
            save_cache(cache)
            print(f"  {index}/{len(pending)} fetched")

    save_cache(cache)
    matched = sum(1 for r in cache.values() if r.get("value_eur_asof") is not None)
    print(
        f"done: {matched}/{len(cache)} players have a market value "
        f"-> {CACHE.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
