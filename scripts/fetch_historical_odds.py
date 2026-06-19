"""
Fetch historical 1X2 odds into data/odds.csv.

Required inputs:
- ODDS_API_KEY environment variable
- ODDS_API_BASE_URL environment variable, e.g. https://example.com
- data/odds_fixture_map.csv with:
  date,home_team,away_team,fixtureId

The API response shown by the user has nested bookmaker/market/outcome records but
does not identify which market/outcome IDs mean home/draw/away. Pass those IDs with
ODDS_HOME_OUTCOME_ID, ODDS_DRAW_OUTCOME_ID, and ODDS_AWAY_OUTCOME_ID.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def latest_price(entries: list[dict]) -> float | None:
    usable = [e for e in entries if e.get("price")]
    if not usable:
        return None
    usable.sort(key=lambda e: e.get("createdAt") or "")
    return float(usable[-1]["price"])


def extract_outcome_price(payload: dict, outcome_id: str, bookmakers: list[str]) -> float | None:
    prices = []
    for bookmaker in bookmakers:
        markets = payload.get("bookmakers", {}).get(bookmaker, {}).get("markets", {})
        for market in markets.values():
            outcome = market.get("outcomes", {}).get(outcome_id)
            if not outcome:
                continue
            for entries in outcome.get("players", {}).values():
                price = latest_price(entries)
                if price:
                    prices.append(price)
    if not prices:
        return None
    return sum(prices) / len(prices)


def fetch_fixture(base_url: str, api_key: str, fixture_id: str, bookmakers: str) -> dict:
    query = urlencode({"fixtureId": fixture_id, "bookmakers": bookmakers})
    url = f"{base_url.rstrip('/')}/v4/historical-odds?{query}"
    req = Request(url, headers={"Authorization": f"Bearer {api_key}", "x-api-key": api_key})
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    api_key = os.environ.get("ODDS_API_KEY")
    base_url = os.environ.get("ODDS_API_BASE_URL")
    bookmakers = os.environ.get("ODDS_BOOKMAKERS", "pinnacle,bet365")
    home_id = os.environ.get("ODDS_HOME_OUTCOME_ID")
    draw_id = os.environ.get("ODDS_DRAW_OUTCOME_ID")
    away_id = os.environ.get("ODDS_AWAY_OUTCOME_ID")

    missing = [
        name
        for name, value in (
            ("ODDS_API_KEY", api_key),
            ("ODDS_API_BASE_URL", base_url),
            ("ODDS_HOME_OUTCOME_ID", home_id),
            ("ODDS_DRAW_OUTCOME_ID", draw_id),
            ("ODDS_AWAY_OUTCOME_ID", away_id),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

    fixture_map = DATA_DIR / "odds_fixture_map.csv"
    if not fixture_map.exists():
        raise SystemExit("Missing data/odds_fixture_map.csv")

    bookmaker_list = [b.strip() for b in bookmakers.split(",") if b.strip()]
    rows = []
    with open(fixture_map, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            payload = fetch_fixture(base_url, api_key, row["fixtureId"], bookmakers)
            home = extract_outcome_price(payload, home_id, bookmaker_list)
            draw = extract_outcome_price(payload, draw_id, bookmaker_list)
            away = extract_outcome_price(payload, away_id, bookmaker_list)
            if home and draw and away:
                rows.append(
                    {
                        "date": row["date"],
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "home_odds": round(home, 4),
                        "draw_odds": round(draw, 4),
                        "away_odds": round(away, 4),
                        "fixtureId": row["fixtureId"],
                        "fetchedAt": datetime.utcnow().isoformat() + "Z",
                    }
                )

    out_path = DATA_DIR / "odds.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "home_team",
                "away_team",
                "home_odds",
                "draw_odds",
                "away_odds",
                "fixtureId",
                "fetchedAt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} odds rows to {out_path}")


if __name__ == "__main__":
    main()
