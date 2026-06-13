"""Download the full EA SPORTS FC ratings database to a trimmed local JSON cache.

Usage: python scripts/download_ea_ratings.py [output_path]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

URL = "https://drop-api.ea.com/rating/ea-sports-fc"
PAGE = 100


def main(output: str = "data/raw/ea_ratings.json") -> None:
    players = []
    offset = 0
    total = None
    session = requests.Session()
    while total is None or offset < total:
        for attempt in range(4):
            try:
                response = session.get(
                    URL,
                    params={"locale": "en", "limit": PAGE, "offset": offset},
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        total = payload["totalItems"]
        for item in payload["items"]:
            players.append(
                {
                    "id": item["id"],
                    "firstName": item.get("firstName") or "",
                    "lastName": item.get("lastName") or "",
                    "commonName": item.get("commonName") or "",
                    "overallRating": item["overallRating"],
                    "birthdate": item.get("birthdate") or "",
                    "position": (item.get("position") or {}).get("shortLabel", ""),
                    "nationality": (item.get("nationality") or {}).get("label", ""),
                    "team": (item.get("team") or {}).get("label", ""),
                    "league": item.get("leagueName") or "",
                    "gender": (item.get("gender") or {}).get("id", 0),
                }
            )
        offset += PAGE
        if offset % 2000 == 0:
            print(f"{offset}/{total}", flush=True)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(players))
    print(f"saved {len(players)} players to {path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
