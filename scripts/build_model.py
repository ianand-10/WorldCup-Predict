"""
Build ELO ratings, ML calibration model, and goalscorer stats from match CSVs.
Outputs JSON files to public/data/ for the Next.js predictor.
"""

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "public" / "data"

INITIAL_ELO = 1500.0
BASE_K = 20.0
HOME_ADV_GOALS = 0.28
MAX_GOALS = 8

TOURNAMENT_WEIGHT = {
    "World Cup": 1.6,
    "World Cup qualification": 1.4,
    "UEFA Euro": 1.5,
    "UEFA Euro qualification": 1.3,
    "Copa América": 1.4,
    "African Cup of Nations": 1.3,
    "AFC Asian Cup": 1.3,
    "CONCACAF Gold Cup": 1.2,
    "UEFA Nations League": 1.25,
    "Friendly": 0.85,
}


def parse_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().upper() in ("TRUE", "1", "YES", "T")


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def goal_expectation(off_rating: float, def_rating: float, venue_boost: float = 0.0) -> float:
    diff = (off_rating - def_rating) / 400.0
    base = math.exp(0.55 * diff + venue_boost - 0.15)
    return max(0.15, min(4.5, base))


def k_factor(tournament: str) -> float:
    for key, weight in TOURNAMENT_WEIGHT.items():
        if key.lower() in str(tournament).lower():
            return BASE_K * weight
    return BASE_K


def outcome_points(home_score: int, away_score: int) -> tuple[float, float]:
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


class TeamRatings:
    def __init__(self):
        self.overall = INITIAL_ELO
        self.offense = INITIAL_ELO
        self.defense = INITIAL_ELO
        self.home_bonus = 0.0
        self.away_penalty = 0.0


def load_matches() -> pd.DataFrame:
    path = DATA_DIR / "matches.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["neutral"] = df["neutral"].apply(parse_bool)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    return df


def load_scorers() -> pd.DataFrame:
    path = DATA_DIR / "goalscorers.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["own_goal"] = df["own_goal"].apply(parse_bool)
    df["penalty"] = df["penalty"].apply(parse_bool)
    return df


def recent_form(results: list[tuple[int, int]], n: int = 5) -> float:
    if not results:
        return 0.5
    pts = []
    for hs, aws in results[-n:]:
        if hs > aws:
            pts.append(1.0)
        elif hs == aws:
            pts.append(0.5)
        else:
            pts.append(0.0)
    return sum(pts) / len(pts)


def compute_elos_and_features(matches: pd.DataFrame):
    teams: dict[str, TeamRatings] = defaultdict(TeamRatings)
    form: dict[str, list[tuple[int, int]]] = defaultdict(list)
    h2h: dict[frozenset, list[int]] = defaultdict(list)
    history: list[dict] = []
    ml_rows: list[dict] = []

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs, aws = int(row["home_score"]), int(row["away_score"])
        neutral = bool(row["neutral"])
        tournament = str(row.get("tournament", "Friendly"))
        k = k_factor(tournament)

        hr, ar = teams[home], teams[away]
        pair = frozenset([home, away])

        h2h_wins = h2h.get(pair, [])
        h2h_home_rate = 0.5
        if h2h_wins:
            h2h_home_rate = sum(h2h_wins) / len(h2h_wins)

        features = {
            "eloOverallDiff": hr.overall - ar.overall,
            "eloOffenseDiff": hr.offense - ar.offense,
            "eloDefenseDiff": hr.defense - ar.defense,
            "isHome": 0.0 if neutral else 1.0,
            "venueAdvantage": (0.0 if neutral else hr.home_bonus - ar.away_penalty),
            "homeForm": recent_form(form[home]),
            "awayForm": recent_form(form[away]),
            "h2hHomeRate": h2h_home_rate,
        }

        outcome = 0 if hs > aws else (2 if hs < aws else 1)
        ml_rows.append({**features, "outcome": outcome})

        venue_boost = 0.0 if neutral else HOME_ADV_GOALS
        exp_home_goals = goal_expectation(hr.offense, ar.defense, venue_boost + hr.home_bonus)
        exp_away_goals = goal_expectation(ar.offense, hr.defense, ar.away_penalty)

        exp_home_result = expected_score(hr.overall, ar.overall)
        if not neutral:
            exp_home_result = expected_score(hr.overall + 65, ar.overall)

        actual_home, actual_away = outcome_points(hs, aws)

        hr.overall += k * (actual_home - exp_home_result)
        ar.overall += k * (actual_away - (1 - exp_home_result))

        off_k = k * 0.85
        def_k = k * 0.85

        hr.offense += off_k * ((hs / max(exp_home_goals, 0.5)) - 1.0) * 0.5
        ar.offense += off_k * ((aws / max(exp_away_goals, 0.5)) - 1.0) * 0.5
        hr.defense += def_k * ((1.0 - aws / max(exp_away_goals, 0.5)) - 0.5) * 0.5
        ar.defense += def_k * ((1.0 - hs / max(exp_home_goals, 0.5)) - 0.5) * 0.5

        if not neutral:
            home_perf = actual_home - exp_home_result
            away_perf = actual_away - (1 - exp_home_result)
            hr.home_bonus += 0.02 * (home_perf - 0.1)
            ar.away_penalty += 0.02 * (away_perf - 0.1)
            hr.home_bonus = max(-0.15, min(0.35, hr.home_bonus))
            ar.away_penalty = max(-0.25, min(0.1, ar.away_penalty))

        form[home].append((hs, aws))
        form[away].append((aws, hs))

        if hs > aws:
            h2h[pair].append(1)
        elif hs < aws:
            h2h[pair].append(0)
        else:
            h2h[pair].append(0.5)

        history.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "home_score": hs,
            "away_score": aws,
            "neutral": neutral,
            "tournament": tournament,
        })

    elos = {}
    for team, r in teams.items():
        elos[team] = {
            "overall": round(r.overall, 1),
            "offense": round(r.offense, 1),
            "defense": round(r.defense, 1),
            "homeBonus": round(r.home_bonus, 3),
            "awayPenalty": round(r.away_penalty, 3),
        }

    return elos, history, ml_rows


def train_ml_model(ml_rows: list[dict]) -> dict | None:
    if len(ml_rows) < 200:
        return None

    feature_names = [
        "eloOverallDiff",
        "eloOffenseDiff",
        "eloDefenseDiff",
        "isHome",
        "venueAdvantage",
        "homeForm",
        "awayForm",
        "h2hHomeRate",
    ]

    X = np.array([[r[f] for f in feature_names] for r in ml_rows])
    y = np.array([r["outcome"] for r in ml_rows])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    model.fit(X_train_s, y_train)
    accuracy = model.score(X_test_s, y_test)
    print(f"ML model accuracy: {accuracy:.3f}")

    return {
        "type": "logistic_regression",
        "featureNames": feature_names,
        "classes": ["home_win", "draw", "away_win"],
        "accuracy": round(float(accuracy), 4),
        "scalerMean": scaler.mean_.tolist(),
        "scalerScale": scaler.scale_.tolist(),
        "coefficients": model.coef_.tolist(),
        "intercepts": model.intercept_.tolist(),
    }


def build_scorer_stats(scorers: pd.DataFrame) -> dict:
    valid = scorers[~scorers["own_goal"]].copy()
    team_goals = valid.groupby("team").size().to_dict()

    result: dict[str, list] = {}
    for team, group in valid.groupby("team"):
        total = team_goals.get(team, 0)
        if total == 0:
            continue
        counts = group["scorer"].value_counts()
        players = []
        for name, count in counts.head(15).items():
            players.append({
                "name": name,
                "goals": int(count),
                "share": round(count / total, 4),
            })
        result[team] = players

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading matches...")
    matches = load_matches()
    print(f"  {len(matches)} matches loaded")

    print("Computing ELO ratings...")
    elos, history, ml_rows = compute_elos_and_features(matches)
    print(f"  {len(elos)} teams rated")

    print("Training ML model...")
    ml_meta = train_ml_model(ml_rows)

    print("Building scorer stats...")
    scorers = load_scorers()
    scorer_stats = build_scorer_stats(scorers)
    print(f"  {len(scorer_stats)} teams with scorer data")

    teams_sorted = sorted(elos.keys())

    model_config = {
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "matchCount": len(matches),
        "teamCount": len(elos),
        "homeAdvantageGoals": HOME_ADV_GOALS,
        "eloBlendWeight": 0.65,
        "mlBlendWeight": 0.35,
        "maxGoals": MAX_GOALS,
        "ml": ml_meta,
    }

    with open(OUT_DIR / "teams.json", "w", encoding="utf-8") as f:
        json.dump({"teams": elos, "teamList": teams_sorted}, f, indent=2)

    with open(OUT_DIR / "scorers.json", "w", encoding="utf-8") as f:
        json.dump({"scorers": scorer_stats}, f, indent=2)

    with open(OUT_DIR / "model.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    print(f"\nDone! Output written to {OUT_DIR}")


if __name__ == "__main__":
    main()
