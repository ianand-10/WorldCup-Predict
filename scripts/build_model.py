"""
Build ELO ratings, ML models, Dixon-Coles params, and live state from match CSVs.
Outputs JSON to public/data/ for the Next.js predictor.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "public" / "data"

INITIAL_ELO = 1500.0
BASE_K = 20.0
HOME_ADV_GOALS = 0.28
MAX_GOALS = 8
DEFAULT_FIFA_POINTS = 1500.0
FIFA_BLEND_WEIGHT = 0.3
ELO_RESULT_FIFA_WEIGHT = 0.35
ELO_BLEND_WEIGHT = 0.55
ML_BLEND_WEIGHT = 0.45
SCORER_HALF_LIFE_DAYS = 365
ML_RECENCY_HALF_LIFE_DAYS = 900

PARAMETER_GRID = [
    {
        "name": "current",
        "baseK": 20.0,
        "homeAdvantageGoals": 0.28,
        "featureFifaBlendWeight": 0.30,
        "eloResultFifaWeight": 0.35,
        "recencyHalfLifeDays": 900,
        "useHistoricalFifa": True,
    },
    {
        "name": "stronger_fifa",
        "baseK": 20.0,
        "homeAdvantageGoals": 0.28,
        "featureFifaBlendWeight": 0.40,
        "eloResultFifaWeight": 0.45,
        "recencyHalfLifeDays": 900,
        "useHistoricalFifa": True,
    },
    {
        "name": "recent_stronger_fifa",
        "baseK": 18.0,
        "homeAdvantageGoals": 0.26,
        "featureFifaBlendWeight": 0.45,
        "eloResultFifaWeight": 0.50,
        "recencyHalfLifeDays": 650,
        "useHistoricalFifa": True,
    },
    {
        "name": "fast_recent",
        "baseK": 24.0,
        "homeAdvantageGoals": 0.30,
        "featureFifaBlendWeight": 0.40,
        "eloResultFifaWeight": 0.50,
        "recencyHalfLifeDays": 600,
        "useHistoricalFifa": True,
    },
    {
        "name": "low_k_fifa",
        "baseK": 16.0,
        "homeAdvantageGoals": 0.24,
        "featureFifaBlendWeight": 0.50,
        "eloResultFifaWeight": 0.55,
        "recencyHalfLifeDays": 750,
        "useHistoricalFifa": True,
    },
    {
        "name": "current_rankings_control",
        "baseK": 20.0,
        "homeAdvantageGoals": 0.28,
        "featureFifaBlendWeight": 0.30,
        "eloResultFifaWeight": 0.35,
        "recencyHalfLifeDays": 900,
        "useHistoricalFifa": False,
    },
    {
        "name": "current_rankings_fast_recent",
        "baseK": 24.0,
        "homeAdvantageGoals": 0.30,
        "featureFifaBlendWeight": 0.40,
        "eloResultFifaWeight": 0.50,
        "recencyHalfLifeDays": 600,
        "useHistoricalFifa": False,
    },
]

FIFA_TEAM_ALIASES = {
    "USA": "United States",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Korea Republic": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Congo DR": "DR Congo",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "DPR Korea": "North Korea",
    "St. Lucia": "Saint Lucia",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "Chinese Taipei": "Taiwan",
    "Kyrgyz Republic": "Kyrgyzstan",
    "The Gambia": "Gambia",
}

FEATURE_NAMES = [
    "eloOverallDiff",
    "eloOffenseDiff",
    "eloDefenseDiff",
    "absEloOverallDiff",
    "isHome",
    "venueAdvantage",
    "homeForm",
    "awayForm",
    "formDiff",
    "homeRecentOpponentFifa",
    "awayRecentOpponentFifa",
    "recentOpponentFifaDiff",
    "h2hHomeRate",
    "homeFifaPoints",
    "awayFifaPoints",
    "fifaPointsDiff",
    "absFifaPointsDiff",
    "fifaEloBlendDiff",
    "blendedOffenseDiff",
    "blendedDefenseDiff",
    "expectedGoalDiff",
    "expectedGoalDiffFifaBlend",
    "closeMatchIndicator",
    "ratingAgreement",
    "fifaEloDisagreement",
]

TOURNAMENT_WEIGHT = {
    "World Cup": 1.75,
    "World Cup qualification": 1.55,
    "FIFA World Cup": 1.75,
    "UEFA Euro": 1.6,
    "UEFA Euro qualification": 1.45,
    "Copa América": 1.55,
    "Copa America": 1.55,
    "African Cup of Nations": 1.45,
    "AFCON": 1.45,
    "AFC Asian Cup": 1.4,
    "CONCACAF Gold Cup": 1.35,
    "CONCACAF Nations League": 1.3,
    "UEFA Nations League": 1.35,
    "Asian Cup qualification": 1.25,
    "CONMEBOL": 1.4,
    "OFC Nations Cup": 1.2,
    "Friendly": 0.7,
    "International Friendly": 0.7,
}


def parse_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().upper() in ("TRUE", "1", "YES", "T")


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def goal_expectation(
    off_rating: float, def_rating: float, venue_boost: float = 0.0
) -> float:
    diff = (off_rating - def_rating) / 400.0
    base = math.exp(0.55 * diff + venue_boost - 0.15)
    return max(0.15, min(4.5, base))


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def dc_tau(i: int, j: int, lh: float, la: float, rho: float) -> float:
    if i == 0 and j == 0:
        return 1.0 - lh * la * rho
    if i == 0 and j == 1:
        return 1.0 + la * rho
    if i == 1 and j == 0:
        return 1.0 + lh * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def tournament_weight(tournament: str) -> float:
    t = str(tournament).lower()
    best = 1.0
    for key, weight in TOURNAMENT_WEIGHT.items():
        if key.lower() in t:
            best = max(best, weight)
    return best


def apply_model_params(params: dict) -> None:
    global BASE_K, HOME_ADV_GOALS, FIFA_BLEND_WEIGHT
    global ELO_RESULT_FIFA_WEIGHT, ML_RECENCY_HALF_LIFE_DAYS

    BASE_K = float(params["baseK"])
    HOME_ADV_GOALS = float(params["homeAdvantageGoals"])
    FIFA_BLEND_WEIGHT = float(params["featureFifaBlendWeight"])
    ELO_RESULT_FIFA_WEIGHT = float(params["eloResultFifaWeight"])
    ML_RECENCY_HALF_LIFE_DAYS = float(params["recencyHalfLifeDays"])


def recency_weight(match_date: pd.Timestamp, reference_date: pd.Timestamp) -> float:
    days_ago = max(0, (reference_date - match_date).days)
    decay = math.log(2) / ML_RECENCY_HALF_LIFE_DAYS
    return math.exp(-days_ago * decay)


def k_factor(tournament: str) -> float:
    return BASE_K * tournament_weight(tournament)


def outcome_points(home_score: int, away_score: int) -> tuple[float, float]:
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


def pair_key(team_a: str, team_b: str) -> str:
    return "|".join(sorted([team_a, team_b]))


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


def recent_average(values: list[float], default: float = DEFAULT_FIFA_POINTS, n: int = 5) -> float:
    if not values:
        return default
    return float(np.mean(values[-n:]))


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


def normalize_fifa_team_name(name: str) -> str:
    cleaned = str(name).strip()
    return FIFA_TEAM_ALIASES.get(cleaned, cleaned)


def parse_fifa_rank_value(raw) -> float | None:
    if pd.isna(raw):
        return None
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    if not digits:
        return None
    return float(digits)


def load_fifa_rankings() -> dict[str, float]:
    candidates = [
        DATA_DIR / "fifa_rankings.csv",
        ROOT / "fifa rankings - Sheet1.csv",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        print("  No FIFA rankings file found — using default points for all teams")
        return {}

    path = max(existing, key=lambda p: sum(1 for _ in open(p, encoding="utf-8")) - 1)

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    team_col = next((c for c in ("team", "country", "nation") if c in df.columns), None)
    if team_col is None:
        print(f"  {path.name} has no team column — skipping FIFA rankings")
        return {}

    points_col = "points" if "points" in df.columns else None
    rank_col = "rank" if "rank" in df.columns else None

    result: dict[str, float] = {}
    for _, row in df.iterrows():
        team = normalize_fifa_team_name(str(row[team_col]).strip())
        if not team:
            continue

        points: float | None = None
        if points_col and pd.notna(row.get(points_col)):
            points = float(row[points_col])
        elif rank_col:
            rank = parse_fifa_rank_value(row.get(rank_col))
            if rank is not None:
                points = max(800.0, 2100.0 - rank * 5.0)

        if points is None:
            points = DEFAULT_FIFA_POINTS

        result[team] = points

    print(f"  {len(result)} FIFA rankings loaded from {path.name}")

    normalized_path = DATA_DIR / "fifa_rankings.csv"
    if path.resolve() != normalized_path.resolve() and result:
        rows = sorted(
            (
                {"team": team, "points": round(points, 2)}
                for team, points in result.items()
            ),
            key=lambda r: r["points"],
            reverse=True,
        )
        try:
            pd.DataFrame(rows).to_csv(normalized_path, index=False)
        except PermissionError:
            print(
                f"  Could not update {normalized_path.name} because it is locked; "
                "continuing with loaded FIFA rankings"
            )

    return result


def load_historical_fifa_rankings() -> dict[tuple[int, int], dict[str, float]]:
    candidates = [
        DATA_DIR / "historical_fifa_rankings.csv",
        ROOT / "historical mens ranking - fifa_mens_rank.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print("  No historical FIFA rankings file found; using current rankings only")
        return {}

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"date", "semester", "team"}
    if not required.issubset(df.columns):
        print(f"  {path.name} is missing historical ranking columns; skipping")
        return {}

    points_col = "total.points" if "total.points" in df.columns else "points"
    if points_col not in df.columns:
        print(f"  {path.name} has no points column; skipping")
        return {}

    snapshots: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for _, row in df.iterrows():
        try:
            year = int(row["date"])
            semester = int(row["semester"])
            points = float(row[points_col])
        except (TypeError, ValueError):
            continue

        team = normalize_fifa_team_name(str(row["team"]).strip())
        if team:
            snapshots[(year, semester)][team] = points

    usable = {key: value for key, value in snapshots.items() if len(value) > 20}
    print(f"  {len(usable)} historical FIFA snapshots loaded from {path.name}")
    return usable


def semester_for_date(match_date: pd.Timestamp) -> tuple[int, int]:
    semester = 1 if int(match_date.month) <= 6 else 2
    return int(match_date.year), semester


def fifa_points_for_date(
    team: str,
    match_date: pd.Timestamp,
    current_fifa: dict[str, float],
    historical_fifa: dict[tuple[int, int], dict[str, float]],
) -> float:
    year, semester = semester_for_date(match_date)
    if year >= 2026:
        return current_fifa.get(team, DEFAULT_FIFA_POINTS)

    available = sorted(k for k in historical_fifa if k <= (year, semester))
    for key in reversed(available):
        points = historical_fifa[key].get(team)
        if points is not None:
            return points

    return current_fifa.get(team, DEFAULT_FIFA_POINTS)


def blend_with_fifa(elo_rating: float, fifa_points: float, weight: float) -> float:
    return (1.0 - weight) * elo_rating + weight * fifa_points


def poisson_outcome_probs(
    lambda_home: float,
    lambda_away: float,
    rho: float,
    max_goals: int = MAX_GOALS,
) -> tuple[float, float, float]:
    win_home = draw = win_away = 0.0
    total = 0.0

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            tau = dc_tau(h, a, lambda_home, lambda_away, rho)
            prob = max(0.0, tau) * poisson_pmf(h, lambda_home) * poisson_pmf(a, lambda_away)
            total += prob
            if h > a:
                win_home += prob
            elif h < a:
                win_away += prob
            else:
                draw += prob

    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3

    return win_home / total, draw / total, win_away / total


def match_lambdas_with_fifa(
    hr: TeamRatings,
    ar: TeamRatings,
    neutral: bool,
    home_fifa: float,
    away_fifa: float,
    fifa_blend: float,
) -> tuple[float, float]:
    home_off = blend_with_fifa(hr.offense, home_fifa, fifa_blend)
    home_def = blend_with_fifa(hr.defense, home_fifa, fifa_blend)
    away_off = blend_with_fifa(ar.offense, away_fifa, fifa_blend)
    away_def = blend_with_fifa(ar.defense, away_fifa, fifa_blend)

    venue_boost = 0.0 if neutral else HOME_ADV_GOALS
    exp_home = goal_expectation(home_off, away_def, venue_boost + hr.home_bonus)
    exp_away = goal_expectation(away_off, home_def, ar.away_penalty)
    return exp_home, exp_away


def h2h_rate_for_team(pair_stats: dict[str, list[float]], team: str) -> float:
    values = pair_stats.get(team, [])
    return float(np.mean(values)) if values else 0.5


def build_match_features(
    home: str,
    away: str,
    neutral: bool,
    hr: TeamRatings,
    ar: TeamRatings,
    form: dict[str, list[tuple[int, int]]],
    opponent_fifa_history: dict[str, list[float]],
    h2h_pair: dict[str, dict[str, list[float]]],
    last_match: dict[str, pd.Timestamp],
    match_date: pd.Timestamp,
    home_fifa: float,
    away_fifa: float,
) -> dict[str, float]:
    pk = pair_key(home, away)
    pair_stats = h2h_pair.get(pk, {})

    venue_boost = 0.0 if neutral else HOME_ADV_GOALS
    exp_home = goal_expectation(hr.offense, ar.defense, venue_boost + hr.home_bonus)
    exp_away = goal_expectation(ar.offense, hr.defense, ar.away_penalty)

    blended_home_overall = blend_with_fifa(hr.overall, home_fifa, FIFA_BLEND_WEIGHT)
    blended_away_overall = blend_with_fifa(ar.overall, away_fifa, FIFA_BLEND_WEIGHT)
    blended_home_off = blend_with_fifa(hr.offense, home_fifa, FIFA_BLEND_WEIGHT)
    blended_home_def = blend_with_fifa(hr.defense, home_fifa, FIFA_BLEND_WEIGHT)
    blended_away_off = blend_with_fifa(ar.offense, away_fifa, FIFA_BLEND_WEIGHT)
    blended_away_def = blend_with_fifa(ar.defense, away_fifa, FIFA_BLEND_WEIGHT)
    exp_home_fifa = goal_expectation(
        blended_home_off, blended_away_def, venue_boost + hr.home_bonus
    )
    exp_away_fifa = goal_expectation(blended_away_off, blended_home_def, ar.away_penalty)

    elo_diff = hr.overall - ar.overall
    fifa_diff = home_fifa - away_fifa
    home_form = recent_form(form[home])
    away_form = recent_form(form[away])
    home_recent_opp_fifa = recent_average(opponent_fifa_history[home])
    away_recent_opp_fifa = recent_average(opponent_fifa_history[away])
    rating_agreement = 1.0 if elo_diff == 0 or fifa_diff == 0 or elo_diff * fifa_diff > 0 else 0.0

    return {
        "eloOverallDiff": elo_diff,
        "eloOffenseDiff": hr.offense - ar.offense,
        "eloDefenseDiff": hr.defense - ar.defense,
        "absEloOverallDiff": abs(elo_diff),
        "isHome": 0.0 if neutral else 1.0,
        "venueAdvantage": 0.0 if neutral else hr.home_bonus - ar.away_penalty,
        "homeForm": home_form,
        "awayForm": away_form,
        "formDiff": home_form - away_form,
        "homeRecentOpponentFifa": home_recent_opp_fifa,
        "awayRecentOpponentFifa": away_recent_opp_fifa,
        "recentOpponentFifaDiff": home_recent_opp_fifa - away_recent_opp_fifa,
        "h2hHomeRate": h2h_rate_for_team(pair_stats, home),
        "homeFifaPoints": home_fifa,
        "awayFifaPoints": away_fifa,
        "fifaPointsDiff": fifa_diff,
        "absFifaPointsDiff": abs(fifa_diff),
        "fifaEloBlendDiff": blended_home_overall - blended_away_overall,
        "blendedOffenseDiff": blended_home_off - blended_away_off,
        "blendedDefenseDiff": blended_home_def - blended_away_def,
        "expectedGoalDiff": exp_home - exp_away,
        "expectedGoalDiffFifaBlend": exp_home_fifa - exp_away_fifa,
        "closeMatchIndicator": 1.0 if abs(exp_home - exp_away) < 0.45 else 0.0,
        "ratingAgreement": rating_agreement,
        "fifaEloDisagreement": 1.0 - rating_agreement,
    }


def compute_elos_and_features(
    matches: pd.DataFrame,
    current_fifa: dict[str, float],
    historical_fifa: dict[tuple[int, int], dict[str, float]],
):
    teams: dict[str, TeamRatings] = defaultdict(TeamRatings)
    form: dict[str, list[tuple[int, int]]] = defaultdict(list)
    opponent_fifa_history: dict[str, list[float]] = defaultdict(list)
    h2h_pair: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    last_match: dict[str, pd.Timestamp] = {}
    history: list[dict] = []
    ml_rows: list[dict] = []
    dc_samples: list[tuple[float, float, int, int]] = []
    reference_date = matches["date"].max()

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs, aws = int(row["home_score"]), int(row["away_score"])
        neutral = bool(row["neutral"])
        tournament = str(row.get("tournament", "Friendly"))
        match_date = row["date"]
        k = k_factor(tournament)
        sample_weight = tournament_weight(tournament) * recency_weight(
            match_date, reference_date
        )
        home_fifa = fifa_points_for_date(home, match_date, current_fifa, historical_fifa)
        away_fifa = fifa_points_for_date(away, match_date, current_fifa, historical_fifa)

        hr, ar = teams[home], teams[away]

        features = build_match_features(
            home,
            away,
            neutral,
            hr,
            ar,
            form,
            opponent_fifa_history,
            h2h_pair,
            last_match,
            match_date,
            home_fifa,
            away_fifa,
        )

        outcome = 0 if hs > aws else (2 if hs < aws else 1)

        exp_home_goals, exp_away_goals = match_lambdas_with_fifa(
            hr, ar, neutral, home_fifa, away_fifa, FIFA_BLEND_WEIGHT
        )

        ml_rows.append(
            {
                **features,
                "outcome": outcome,
                "isDraw": 1 if hs == aws else 0,
                "homeWinNotDraw": 1 if hs > aws else 0,
                "sampleWeight": sample_weight,
                "expHomeGoals": exp_home_goals,
                "expAwayGoals": exp_away_goals,
                "isNeutral": neutral,
                "homeOffense": hr.offense,
                "homeDefense": hr.defense,
                "awayOffense": ar.offense,
                "awayDefense": ar.defense,
                "homeBonus": hr.home_bonus,
                "awayPenalty": ar.away_penalty,
                "homeFifa": home_fifa,
                "awayFifa": away_fifa,
            }
        )

        dc_samples.append((exp_home_goals, exp_away_goals, hs, aws))

        home_result_rating = blend_with_fifa(
            hr.overall, home_fifa, ELO_RESULT_FIFA_WEIGHT
        )
        away_result_rating = blend_with_fifa(
            ar.overall, away_fifa, ELO_RESULT_FIFA_WEIGHT
        )
        exp_home_result = expected_score(home_result_rating, away_result_rating)
        if not neutral:
            exp_home_result = expected_score(home_result_rating + 65, away_result_rating)

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
        opponent_fifa_history[home].append(away_fifa)
        opponent_fifa_history[away].append(home_fifa)

        pk = pair_key(home, away)
        home_result = 1.0 if hs > aws else (0.5 if hs == aws else 0.0)
        away_result = 1.0 if aws > hs else (0.5 if hs == aws else 0.0)
        h2h_pair[pk][home].append(home_result)
        h2h_pair[pk][away].append(away_result)

        last_match[home] = match_date
        last_match[away] = match_date

        history.append(
            {
                "date": match_date.strftime("%Y-%m-%d"),
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": aws,
                "neutral": neutral,
                "tournament": tournament,
            }
        )

    elos = {}
    for team, r in teams.items():
        elos[team] = {
            "overall": round(r.overall, 1),
            "offense": round(r.offense, 1),
            "defense": round(r.defense, 1),
            "homeBonus": round(r.home_bonus, 3),
            "awayPenalty": round(r.away_penalty, 3),
        }

    live_form = {team: round(recent_form(form[team]), 4) for team in teams}
    live_recent_opponent_fifa = {
        team: round(recent_average(opponent_fifa_history[team]), 1) for team in teams
    }
    live_last_match = {
        team: last_match[team].strftime("%Y-%m-%d") for team in last_match
    }
    live_h2h: dict[str, dict] = {}
    for pk, stats in h2h_pair.items():
        t1, t2 = pk.split("|")
        live_h2h[pk] = {
            t1: round(h2h_rate_for_team(stats, t1), 4),
            t2: round(h2h_rate_for_team(stats, t2), 4),
            "matches": max(len(stats.get(t1, [])), len(stats.get(t2, []))),
        }

    live_state = {
        "referenceDate": reference_date.strftime("%Y-%m-%d"),
        "form": live_form,
        "recentOpponentFifa": live_recent_opponent_fifa,
        "lastMatchDate": live_last_match,
        "h2h": live_h2h,
        "fifaPoints": {
            team: round(current_fifa.get(team, DEFAULT_FIFA_POINTS), 1) for team in teams
        },
    }
    for team, points in current_fifa.items():
        live_state["fifaPoints"][team] = round(points, 1)

    return elos, history, ml_rows, dc_samples, live_state


def estimate_dixon_coles_rho(samples: list[tuple[float, float, int, int]]) -> float:
    if len(samples) < 100:
        return -0.13

    subset = samples[-3000:]
    best_rho = -0.13
    best_ll = -float("inf")

    for rho in np.linspace(-0.25, -0.01, 25):
        ll = 0.0
        for lh, la, hs, aws in subset:
            tau = dc_tau(hs, aws, lh, la, rho)
            if tau <= 0:
                continue
            p = tau * poisson_pmf(hs, lh) * poisson_pmf(aws, la)
            if p > 0:
                ll += math.log(p)
        if ll > best_ll:
            best_ll = ll
            best_rho = float(rho)

    return round(best_rho, 4)


def fit_multinomial_with_draw_calibration(
    X_scaled: np.ndarray,
    y_outcome: np.ndarray,
    y_draw: np.ndarray,
    weights: np.ndarray,
) -> tuple[LogisticRegression, LogisticRegression]:
    outcome_model = LogisticRegression(
        max_iter=1500,
        solver="lbfgs",
        C=0.9,
        random_state=42,
    )
    draw_model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        C=0.7,
        random_state=42,
    )
    outcome_model.fit(X_scaled, y_outcome, sample_weight=weights)
    draw_model.fit(X_scaled, y_draw, sample_weight=weights)
    return outcome_model, draw_model


def fit_random_forest_with_draw_calibration(
    X_scaled: np.ndarray,
    y_outcome: np.ndarray,
    y_draw: np.ndarray,
    weights: np.ndarray,
) -> tuple[RandomForestClassifier, LogisticRegression]:
    outcome_model = RandomForestClassifier(
        n_estimators=140,
        max_depth=7,
        min_samples_leaf=18,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )
    draw_model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        C=0.7,
        random_state=42,
    )
    outcome_model.fit(X_scaled, y_outcome, sample_weight=weights)
    draw_model.fit(X_scaled, y_draw, sample_weight=weights)
    return outcome_model, draw_model


def fit_extra_trees_with_draw_calibration(
    X_scaled: np.ndarray,
    y_outcome: np.ndarray,
    y_draw: np.ndarray,
    weights: np.ndarray,
) -> tuple[ExtraTreesClassifier, LogisticRegression]:
    outcome_model = ExtraTreesClassifier(
        n_estimators=180,
        max_depth=8,
        min_samples_leaf=14,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    draw_model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        C=0.7,
        random_state=42,
    )
    outcome_model.fit(X_scaled, y_outcome, sample_weight=weights)
    draw_model.fit(X_scaled, y_draw, sample_weight=weights)
    return outcome_model, draw_model


def fit_xgboost_with_draw_calibration(
    X_scaled: np.ndarray,
    y_outcome: np.ndarray,
    y_draw: np.ndarray,
    weights: np.ndarray,
):
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed")

    outcome_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=180,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=8,
        reg_lambda=2.5,
        reg_alpha=0.2,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
    )
    draw_model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        C=0.7,
        random_state=42,
    )
    outcome_model.fit(X_scaled, y_outcome, sample_weight=weights)
    draw_model.fit(X_scaled, y_draw, sample_weight=weights)
    return outcome_model, draw_model


def fit_lightgbm_with_draw_calibration(
    X_scaled: np.ndarray,
    y_outcome: np.ndarray,
    y_draw: np.ndarray,
    weights: np.ndarray,
):
    if LGBMClassifier is None:
        raise ImportError("lightgbm is not installed")

    outcome_model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=220,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=35,
        reg_lambda=2.0,
        reg_alpha=0.2,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )
    draw_model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        C=0.7,
        random_state=42,
    )
    outcome_model.fit(X_scaled, y_outcome, sample_weight=weights)
    draw_model.fit(X_scaled, y_draw, sample_weight=weights)
    return outcome_model, draw_model


def blended_predict_proba(
    outcome_model,
    draw_model: LogisticRegression,
    X: np.ndarray,
    draw_blend: float = 0.35,
) -> np.ndarray:
    base = outcome_model.predict_proba(X)
    p_draw_specialist = draw_model.predict_proba(X)[:, 1]

    blended = base.copy()
    blended[:, 1] = (1.0 - draw_blend) * base[:, 1] + draw_blend * p_draw_specialist

    row_sums = blended.sum(axis=1, keepdims=True)
    return blended / row_sums


def outcome_accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    preds = probs.argmax(axis=1)
    return float((preds == y_true).mean())


def apply_decision_calibration(probs: np.ndarray, calibration: dict | None) -> np.ndarray:
    if not calibration:
        return probs

    adjusted = probs.copy()
    draw_multiplier = float(calibration.get("drawMultiplier", 1.0))
    close_margin = float(calibration.get("closeMargin", 0.0))
    min_draw_probability = float(calibration.get("minDrawProbability", 0.0))
    favorite_multiplier = float(calibration.get("favoriteMultiplier", 1.0))

    win_gap = np.abs(adjusted[:, 0] - adjusted[:, 2])
    favorite_probs = np.maximum(adjusted[:, 0], adjusted[:, 2])
    close_mask = (win_gap <= close_margin) & (adjusted[:, 1] >= min_draw_probability)
    favorite_mask = ~close_mask

    adjusted[close_mask, 1] *= draw_multiplier
    if favorite_multiplier != 1.0:
        home_favorite = favorite_mask & (adjusted[:, 0] >= adjusted[:, 2])
        away_favorite = favorite_mask & (adjusted[:, 2] > adjusted[:, 0])
        confident = favorite_probs >= float(calibration.get("favoriteMinProbability", 0.0))
        adjusted[home_favorite & confident, 0] *= favorite_multiplier
        adjusted[away_favorite & confident, 2] *= favorite_multiplier

    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def tune_decision_calibration(y_true: np.ndarray, probs: np.ndarray) -> tuple[dict, np.ndarray]:
    baseline_acc = outcome_accuracy(y_true, probs)
    baseline_loss = log_loss(y_true, probs, labels=[0, 1, 2])
    best = {
        "drawMultiplier": 1.0,
        "closeMargin": 0.0,
        "minDrawProbability": 0.0,
        "favoriteMultiplier": 1.0,
        "favoriteMinProbability": 0.0,
        "accuracy": round(baseline_acc, 4),
        "logLoss": round(float(baseline_loss), 4),
    }
    best_probs = probs
    best_key = (baseline_acc, -baseline_loss)

    for draw_multiplier in (1.0, 1.08, 1.16, 1.24, 1.32, 1.4, 1.55):
        for close_margin in (0.02, 0.04, 0.06, 0.08, 0.1, 0.14, 0.18, 0.24):
            for min_draw_probability in (0.16, 0.18, 0.2, 0.22, 0.24, 0.26):
                for favorite_multiplier in (1.0, 1.04, 1.08):
                    calibration = {
                        "drawMultiplier": draw_multiplier,
                        "closeMargin": close_margin,
                        "minDrawProbability": min_draw_probability,
                        "favoriteMultiplier": favorite_multiplier,
                        "favoriteMinProbability": 0.42,
                    }
                    calibrated = apply_decision_calibration(probs, calibration)
                    acc = outcome_accuracy(y_true, calibrated)
                    loss = log_loss(y_true, calibrated, labels=[0, 1, 2])
                    key = (acc, -loss)
                    if key > best_key:
                        best_key = key
                        best_probs = calibrated
                        best = {
                            **calibration,
                            "accuracy": round(float(acc), 4),
                            "logLoss": round(float(loss), 4),
                        }

    best["baselineAccuracy"] = round(float(baseline_acc), 4)
    best["baselineLogLoss"] = round(float(baseline_loss), 4)
    return best, best_probs


def serialize_binary_logistic(model: LogisticRegression) -> dict:
    return {
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
    }


def serialize_multinomial(model: LogisticRegression) -> dict:
    return {
        "coefficients": model.coef_.tolist(),
        "intercepts": model.intercept_.tolist(),
    }


def serialize_tree(tree) -> dict:
    sklearn_tree = tree.tree_
    return {
        "childrenLeft": sklearn_tree.children_left.tolist(),
        "childrenRight": sklearn_tree.children_right.tolist(),
        "feature": sklearn_tree.feature.tolist(),
        "threshold": sklearn_tree.threshold.tolist(),
        "value": sklearn_tree.value[:, 0, :].tolist(),
    }


def serialize_tree_ensemble(model) -> dict:
    return {
        "classes": [int(c) for c in model.classes_.tolist()],
        "trees": [serialize_tree(tree) for tree in model.estimators_],
    }


def forest_predict_proba(model_meta: dict, X: np.ndarray) -> np.ndarray:
    trees = model_meta["outcomeModel"]["trees"]
    classes = model_meta["outcomeModel"].get("classes", [0, 1, 2])
    class_to_index = {int(cls): idx for idx, cls in enumerate(classes)}
    probs = np.zeros((X.shape[0], 3))

    for tree in trees:
        left = tree["childrenLeft"]
        right = tree["childrenRight"]
        feature = tree["feature"]
        threshold = tree["threshold"]
        value = tree["value"]

        for i, row in enumerate(X):
            node = 0
            while left[node] != -1:
                node = left[node] if row[feature[node]] <= threshold[node] else right[node]
            counts = np.array(value[node], dtype=float)
            total = counts.sum()
            if total <= 0:
                continue
            for raw_idx, cls in enumerate(classes):
                probs[i, class_to_index[int(cls)]] += counts[raw_idx] / total

    probs /= max(1, len(trees))
    row_sums = probs.sum(axis=1, keepdims=True)
    return probs / np.where(row_sums == 0, 1, row_sums)


def ml_probs_from_meta(ml_meta: dict, rows: list[dict]) -> np.ndarray:
    X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows])
    mean = np.array(ml_meta["scalerMean"])
    scale = np.array(ml_meta["scalerScale"])
    X_scaled = (X - mean) / np.where(scale == 0, 1, scale)

    if ml_meta.get("type") in (
        "random_forest_with_draw_calibration",
        "extra_trees_with_draw_calibration",
    ):
        base = forest_predict_proba(ml_meta, X_scaled)
    elif ml_meta.get("type") == "stacking_logistic":
        stack_parts = []
        for base_model in ml_meta["outcomeModel"]["baseModels"]:
            base_meta = {
                "type": base_model["type"],
                "outcomeModel": base_model["model"],
                "drawModel": ml_meta["drawModel"],
                "drawBlendWeight": 0.0,
                "scalerMean": ml_meta["scalerMean"],
                "scalerScale": ml_meta["scalerScale"],
            }
            stack_parts.append(ml_probs_from_meta(base_meta, rows))
        meta_X = np.concatenate(stack_parts, axis=1)
        coef = np.array(ml_meta["outcomeModel"]["metaModel"]["coefficients"])
        intercepts = np.array(ml_meta["outcomeModel"]["metaModel"]["intercepts"])
        logits = intercepts + meta_X @ coef.T
        logits -= logits.max(axis=1, keepdims=True)
        base = np.exp(logits)
        base /= base.sum(axis=1, keepdims=True)
    else:
        coef = np.array(ml_meta["outcomeModel"]["coefficients"])
        intercepts = np.array(ml_meta["outcomeModel"]["intercepts"])
        logits = intercepts + X_scaled @ coef.T
        logits -= logits.max(axis=1, keepdims=True)
        base = np.exp(logits)
        base /= base.sum(axis=1, keepdims=True)

    draw_blend = ml_meta.get("drawBlendWeight", 0.35)
    draw_coef = np.array(ml_meta["drawModel"]["coefficients"])
    draw_intercept = ml_meta["drawModel"]["intercept"]
    draw_logit = draw_intercept + X_scaled @ draw_coef
    p_draw_specialist = 1.0 / (1.0 + np.exp(-draw_logit))

    ml_probs = base.copy()
    ml_probs[:, 1] = (1.0 - draw_blend) * base[:, 1] + draw_blend * p_draw_specialist
    ml_probs /= ml_probs.sum(axis=1, keepdims=True)
    return ml_probs


def candidate_predict_proba(kind: str, model, X: np.ndarray) -> np.ndarray:
    if kind in (
        "multinomial_with_draw_calibration",
        "random_forest_with_draw_calibration",
        "extra_trees_with_draw_calibration",
        "xgboost_with_draw_calibration",
        "lightgbm_with_draw_calibration",
    ):
        return model.predict_proba(X)
    raise ValueError(f"Unsupported candidate model: {kind}")


def fit_candidate_model(kind: str, X, y_outcome, y_draw, weights):
    if kind == "random_forest_with_draw_calibration":
        return fit_random_forest_with_draw_calibration(X, y_outcome, y_draw, weights)
    if kind == "extra_trees_with_draw_calibration":
        return fit_extra_trees_with_draw_calibration(X, y_outcome, y_draw, weights)
    if kind == "xgboost_with_draw_calibration":
        return fit_xgboost_with_draw_calibration(X, y_outcome, y_draw, weights)
    if kind == "lightgbm_with_draw_calibration":
        return fit_lightgbm_with_draw_calibration(X, y_outcome, y_draw, weights)
    return fit_multinomial_with_draw_calibration(X, y_outcome, y_draw, weights)


DEPLOYABLE_KINDS = [
    "multinomial_with_draw_calibration",
    "random_forest_with_draw_calibration",
    "extra_trees_with_draw_calibration",
]

BOOSTER_KINDS = [
    kind
    for kind, available in (
        ("xgboost_with_draw_calibration", XGBClassifier is not None),
        ("lightgbm_with_draw_calibration", LGBMClassifier is not None),
    )
    if available
]

STACK_BASE_KINDS = DEPLOYABLE_KINDS
BOOSTER_STACK_BASE_KINDS = [*DEPLOYABLE_KINDS, *BOOSTER_KINDS]


def train_stacking_model(
    X: np.ndarray,
    y_outcome: np.ndarray,
    weights: np.ndarray,
    base_kinds: list[str] | None = None,
) -> dict:
    base_kinds = base_kinds or STACK_BASE_KINDS
    tscv = TimeSeriesSplit(n_splits=5)
    meta_X = np.zeros((len(X), len(base_kinds) * 3))
    covered = np.zeros(len(X), dtype=bool)

    for train_idx, valid_idx in tscv.split(X):
        if len(train_idx) < 800 or len(valid_idx) < 50:
            continue

        for base_i, kind in enumerate(base_kinds):
            outcome_model, _ = fit_candidate_model(
                kind,
                X[train_idx],
                y_outcome[train_idx],
                np.array(y_outcome[train_idx] == 1, dtype=int),
                weights[train_idx],
            )
            probs = candidate_predict_proba(kind, outcome_model, X[valid_idx])
            meta_X[valid_idx, base_i * 3 : base_i * 3 + 3] = probs

        covered[valid_idx] = True

    if covered.sum() < 500:
        raise ValueError("Not enough out-of-fold rows to train stacking model")

    meta_model = LogisticRegression(
        max_iter=1500,
        solver="lbfgs",
        C=0.7,
        random_state=42,
    )
    meta_model.fit(meta_X[covered], y_outcome[covered], sample_weight=weights[covered])

    base_models = []
    for kind in base_kinds:
        outcome_model, _ = fit_candidate_model(
            kind,
            X,
            y_outcome,
            np.array(y_outcome == 1, dtype=int),
            weights,
        )
        base_models.append((kind, outcome_model))

    return {
        "baseModels": base_models,
        "metaModel": meta_model,
    }


def stacking_predict_proba(stack_model: dict, X: np.ndarray) -> np.ndarray:
    meta_X = np.zeros((X.shape[0], len(stack_model["baseModels"]) * 3))
    for base_i, (kind, model) in enumerate(stack_model["baseModels"]):
        probs = candidate_predict_proba(kind, model, X)
        meta_X[:, base_i * 3 : base_i * 3 + 3] = probs

    return stack_model["metaModel"].predict_proba(meta_X)


def serialize_outcome_model(kind: str, model) -> dict:
    if kind == "multinomial_with_draw_calibration":
        return serialize_multinomial(model)
    if kind in ("random_forest_with_draw_calibration", "extra_trees_with_draw_calibration"):
        return serialize_tree_ensemble(model)
    if kind == "stacking_logistic":
        return {
            "baseModels": [
                {"type": base_kind, "model": serialize_outcome_model(base_kind, base_model)}
                for base_kind, base_model in model["baseModels"]
            ],
            "metaModel": serialize_multinomial(model["metaModel"]),
        }
    raise ValueError(f"Unsupported model type: {kind}")


def select_draw_blend(
    outcome_model,
    draw_model: LogisticRegression,
    X_test: np.ndarray,
    y_true: np.ndarray,
    candidates: tuple[float, ...] = (0.0, 0.15, 0.25, 0.35, 0.45, 0.55),
) -> tuple[float, np.ndarray, float, float]:
    best_blend = candidates[0]
    best_probs = blended_predict_proba(outcome_model, draw_model, X_test, best_blend)
    best_acc = outcome_accuracy(y_true, best_probs)
    best_loss = log_loss(y_true, best_probs, labels=[0, 1, 2])

    for draw_blend in candidates[1:]:
        probs = blended_predict_proba(outcome_model, draw_model, X_test, draw_blend)
        acc = outcome_accuracy(y_true, probs)
        loss = log_loss(y_true, probs, labels=[0, 1, 2])
        if (acc, -loss) > (best_acc, -best_loss):
            best_blend = draw_blend
            best_probs = probs
            best_acc = acc
            best_loss = loss

    return best_blend, best_probs, float(best_acc), float(best_loss)


def train_ml_models(ml_rows: list[dict]) -> dict | None:
    if len(ml_rows) < 300:
        return None

    X = np.array([[r[f] for f in FEATURE_NAMES] for r in ml_rows])
    y_outcome = np.array([r["outcome"] for r in ml_rows])
    y_draw = np.array([r["isDraw"] for r in ml_rows])
    weights = np.array([r["sampleWeight"] for r in ml_rows])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)
    candidates = [*DEPLOYABLE_KINDS, *BOOSTER_KINDS]
    candidate_results: dict[str, dict[str, list[float] | float]] = {
        kind: {"cvAccuracy": [], "cvLogLoss": []} for kind in candidates
    }

    for train_idx, test_idx in tscv.split(X_scaled):
        if len(train_idx) < 800 or len(test_idx) < 50:
            continue

        fold_scaler = StandardScaler()
        X_train = fold_scaler.fit_transform(X[train_idx])
        X_test = fold_scaler.transform(X[test_idx])

        for kind in candidates:
            fold_outcome, fold_draw = fit_candidate_model(
                kind,
                X_train,
                y_outcome[train_idx],
                y_draw[train_idx],
                weights[train_idx],
            )
            draw_blend = 0.25 if kind == "random_forest_with_draw_calibration" else 0.35
            probs = blended_predict_proba(fold_outcome, fold_draw, X_test, draw_blend)
            candidate_results[kind]["cvAccuracy"].append(
                outcome_accuracy(y_outcome[test_idx], probs)
            )
            candidate_results[kind]["cvLogLoss"].append(
                log_loss(y_outcome[test_idx], probs, labels=[0, 1, 2])
            )

    split_idx = int(len(X_scaled) * 0.85)
    holdout_scaler = StandardScaler()
    X_train_h = holdout_scaler.fit_transform(X[:split_idx])
    X_test_h = holdout_scaler.transform(X[split_idx:])

    best_kind = DEPLOYABLE_KINDS[0]
    best_holdout_acc = -1.0
    best_holdout_loss = float("inf")
    best_draw_blend = 0.35
    holdout_summary: dict[str, dict[str, float]] = {}

    holdout_candidates = [*candidates, "stacking_logistic"]
    if BOOSTER_KINDS:
        holdout_candidates.append("stacking_logistic_with_boosters")
    for kind in holdout_candidates:
        if kind in ("stacking_logistic", "stacking_logistic_with_boosters"):
            base_kinds = (
                BOOSTER_STACK_BASE_KINDS
                if kind == "stacking_logistic_with_boosters"
                else STACK_BASE_KINDS
            )
            stack_model = train_stacking_model(
                X_train_h,
                y_outcome[:split_idx],
                weights[:split_idx],
                base_kinds,
            )
            holdout_probs = stacking_predict_proba(stack_model, X_test_h)
            holdout_acc = outcome_accuracy(y_outcome[split_idx:], holdout_probs)
            holdout_loss = log_loss(
                y_outcome[split_idx:], holdout_probs, labels=[0, 1, 2]
            )
            draw_blend = 0.0
        else:
            holdout_outcome, holdout_draw = fit_candidate_model(
                kind,
                X_train_h,
                y_outcome[:split_idx],
                y_draw[:split_idx],
                weights[:split_idx],
            )
            draw_blend, holdout_probs, holdout_acc, holdout_loss = select_draw_blend(
                holdout_outcome,
                holdout_draw,
                X_test_h,
                y_outcome[split_idx:],
            )
        holdout_summary[kind] = {
            "accuracy": float(holdout_acc),
            "logLoss": float(holdout_loss),
            "drawBlend": float(draw_blend),
        }
        is_deployable = kind in (*DEPLOYABLE_KINDS, "stacking_logistic")
        if is_deployable and (holdout_acc, -holdout_loss) > (
            best_holdout_acc,
            -best_holdout_loss,
        ):
            best_kind = kind
            best_holdout_acc = float(holdout_acc)
            best_holdout_loss = float(holdout_loss)
            best_draw_blend = float(draw_blend)

    if best_kind == "stacking_logistic":
        outcome_model = train_stacking_model(X_scaled, y_outcome, weights, STACK_BASE_KINDS)
        draw_model = LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            C=0.7,
            random_state=42,
        )
        draw_model.fit(X_scaled, y_draw, sample_weight=weights)
    else:
        outcome_model, draw_model = fit_candidate_model(
            best_kind, X_scaled, y_outcome, y_draw, weights
        )

    selected_scores = candidate_results.get(best_kind, {}).get("cvAccuracy", [])
    cv_mean = float(np.mean(selected_scores)) if selected_scores else best_holdout_acc
    cv_std = float(np.std(selected_scores)) if selected_scores else 0.0

    print("  ML candidate holdout results:")
    for kind, metrics in holdout_summary.items():
        print(
            f"    {kind}: acc {metrics['accuracy']:.3f}, "
            f"log loss {metrics['logLoss']:.3f}, draw blend {metrics['drawBlend']:.2f}"
        )
    print(f"  Selected ML model: {best_kind}")
    print(f"  ML time-series CV accuracy: {cv_mean:.3f} (+/- {cv_std:.3f})")
    print(f"  ML chronological holdout (last 15%): {best_holdout_acc:.3f}")

    outcome_payload = serialize_outcome_model(best_kind, outcome_model)

    return {
        "type": best_kind,
        "featureNames": FEATURE_NAMES,
        "classes": ["home_win", "draw", "away_win"],
        "accuracy": round(cv_mean, 4),
        "accuracyStd": round(cv_std, 4),
        "holdoutAccuracy": round(float(best_holdout_acc), 4),
        "holdoutLogLoss": round(float(best_holdout_loss), 4),
        "cvFolds": len(selected_scores),
        "drawBlendWeight": best_draw_blend,
        "candidateResults": {
            kind: {
                "cvAccuracy": round(float(np.mean(values["cvAccuracy"])), 4)
                if values["cvAccuracy"]
                else None,
                "cvLogLoss": round(float(np.mean(values["cvLogLoss"])), 4)
                if values["cvLogLoss"]
                else None,
                "holdoutAccuracy": round(holdout_summary[kind]["accuracy"], 4),
                "holdoutLogLoss": round(holdout_summary[kind]["logLoss"], 4),
                "drawBlendWeight": round(holdout_summary[kind]["drawBlend"], 4),
            }
            for kind, values in candidate_results.items()
        },
        "holdoutCandidateResults": {
            kind: {
                "holdoutAccuracy": round(metrics["accuracy"], 4),
                "holdoutLogLoss": round(metrics["logLoss"], 4),
                "drawBlendWeight": round(metrics["drawBlend"], 4),
            }
            for kind, metrics in holdout_summary.items()
        },
        "scalerMean": scaler.mean_.tolist(),
        "scalerScale": scaler.scale_.tolist(),
        "outcomeModel": outcome_payload,
        "drawModel": serialize_binary_logistic(draw_model),
    }


def lambdas_from_row(row: dict, fifa_blend: float) -> tuple[float, float]:
    neutral = bool(row["isNeutral"])
    home_off = blend_with_fifa(row["homeOffense"], row["homeFifa"], fifa_blend)
    home_def = blend_with_fifa(row["homeDefense"], row["homeFifa"], fifa_blend)
    away_off = blend_with_fifa(row["awayOffense"], row["awayFifa"], fifa_blend)
    away_def = blend_with_fifa(row["awayDefense"], row["awayFifa"], fifa_blend)
    venue_boost = 0.0 if neutral else HOME_ADV_GOALS
    exp_home = goal_expectation(home_off, away_def, venue_boost + row["homeBonus"])
    exp_away = goal_expectation(away_off, home_def, row["awayPenalty"])
    return exp_home, exp_away


def evaluate_system_accuracy(
    ml_rows: list[dict],
    ml_meta: dict | None,
    dixon_coles_rho: float,
    fifa_blend: float,
    elo_blend: float,
    ml_blend: float,
    decision_calibration: dict | None = None,
) -> dict[str, float]:
    split_idx = int(len(ml_rows) * 0.85)
    holdout = ml_rows[split_idx:]
    if len(holdout) < 50:
        return {}

    y_true = np.array([r["outcome"] for r in holdout])
    poisson_probs = np.zeros((len(holdout), 3))
    ml_probs = np.zeros((len(holdout), 3))
    ensemble_probs = np.zeros((len(holdout), 3))

    if ml_meta:
        ml_probs = ml_probs_from_meta(ml_meta, holdout)

    for i, row in enumerate(holdout):
        exp_home, exp_away = lambdas_from_row(row, fifa_blend)
        wh, dr, wa = poisson_outcome_probs(exp_home, exp_away, dixon_coles_rho)
        poisson_probs[i] = [wh, dr, wa]

        if ml_meta:
            ensemble_probs[i] = elo_blend * poisson_probs[i] + ml_blend * ml_probs[i]
            ensemble_probs[i] /= ensemble_probs[i].sum()
        else:
            ensemble_probs[i] = poisson_probs[i]

    calibrated_probs = apply_decision_calibration(ensemble_probs, decision_calibration)

    return {
        "poissonAccuracy": round(outcome_accuracy(y_true, poisson_probs), 4),
        "mlAccuracyHoldout": round(outcome_accuracy(y_true, ml_probs), 4) if ml_meta else None,
        "combinedAccuracy": round(outcome_accuracy(y_true, ensemble_probs), 4),
        "calibratedAccuracy": round(outcome_accuracy(y_true, calibrated_probs), 4),
        "holdoutSize": len(holdout),
    }


def tune_blend_weights(
    ml_rows: list[dict],
    ml_meta: dict | None,
    dixon_coles_rho: float,
) -> tuple[float, float, float, dict]:
    if not ml_meta or len(ml_rows) < 500:
        return FIFA_BLEND_WEIGHT, ELO_BLEND_WEIGHT, ML_BLEND_WEIGHT

    split_idx = int(len(ml_rows) * 0.85)
    holdout = ml_rows[split_idx:]
    y_true = np.array([r["outcome"] for r in holdout])

    best_fifa = FIFA_BLEND_WEIGHT
    best_elo = ELO_BLEND_WEIGHT
    best_ml = ML_BLEND_WEIGHT
    best_acc = -1.0
    best_loss = float("inf")
    best_ensemble: np.ndarray | None = None

    ml_probs = ml_probs_from_meta(ml_meta, holdout)

    poisson_by_fifa: dict[float, np.ndarray] = {}
    for fifa_blend in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65):
        probs = np.zeros((len(holdout), 3))
        for i, row in enumerate(holdout):
            exp_home, exp_away = lambdas_from_row(row, fifa_blend)
            wh, dr, wa = poisson_outcome_probs(exp_home, exp_away, dixon_coles_rho)
            probs[i] = [wh, dr, wa]
        poisson_by_fifa[fifa_blend] = probs

    for fifa_blend, poisson_probs in poisson_by_fifa.items():
        for elo_w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
            ml_w = 1.0 - elo_w
            ensemble = elo_w * poisson_probs + ml_w * ml_probs
            ensemble /= ensemble.sum(axis=1, keepdims=True)
            acc = outcome_accuracy(y_true, ensemble)
            loss = log_loss(y_true, ensemble, labels=[0, 1, 2])
            if (acc, -loss) > (best_acc, -best_loss):
                best_acc = acc
                best_loss = loss
                best_fifa = fifa_blend
                best_elo = elo_w
                best_ml = ml_w
                best_ensemble = ensemble

    best_calibration, calibrated = tune_decision_calibration(y_true, best_ensemble)
    calibrated_acc = outcome_accuracy(y_true, calibrated)
    print(
        f"  Tuned blends — FIFA: {best_fifa:.2f}, Poisson: {best_elo:.2f}, ML: {best_ml:.2f} "
        f"(raw holdout acc {best_acc:.3f}, calibrated {calibrated_acc:.3f})"
    )
    if best_calibration:
        print(
            f"  Decision calibration — draw x{best_calibration['drawMultiplier']:.2f}, "
            f"close margin {best_calibration['closeMargin']:.2f}, "
            f"min draw {best_calibration['minDrawProbability']:.2f}"
        )
    return best_fifa, best_elo, best_ml, best_calibration


def build_scorer_stats(scorers: pd.DataFrame) -> dict:
    valid = scorers[~scorers["own_goal"]].copy()
    reference_date = valid["date"].max()
    recent_cutoff = reference_date - pd.Timedelta(days=365)
    decay = math.log(2) / SCORER_HALF_LIFE_DAYS

    result: dict[str, list] = {}
    for team, group in valid.groupby("team"):
        weighted: dict[str, float] = defaultdict(float)
        recent_counts: dict[str, int] = defaultdict(int)
        total_weight = 0.0

        for _, goal in group.iterrows():
            days_ago = max(0, (reference_date - goal["date"]).days)
            weight = math.exp(-days_ago * decay)
            weighted[goal["scorer"]] += weight
            total_weight += weight
            if goal["date"] >= recent_cutoff:
                recent_counts[goal["scorer"]] += 1

        if total_weight <= 0:
            continue

        players = []
        for name, score in sorted(weighted.items(), key=lambda x: x[1], reverse=True)[:15]:
            players.append(
                {
                    "name": name,
                    "goals": recent_counts.get(name, 0),
                    "weightedGoals": round(score, 2),
                    "share": round(score / total_weight, 4),
                }
            )
        result[team] = players

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading matches...")
    matches = load_matches()
    print(f"  {len(matches)} matches loaded")

    print("Loading FIFA rankings...")
    current_fifa = load_fifa_rankings()
    historical_fifa = load_historical_fifa_rankings()
    best_run = None
    experiment_results = []

    for params in PARAMETER_GRID:
        apply_model_params(params)
        print(f"\nExperiment: {params['name']}")

        print("  Computing ELO ratings and features...")
        fifa_history_for_run = historical_fifa if params.get("useHistoricalFifa", True) else {}
        elos, history, ml_rows, dc_samples, live_state = compute_elos_and_features(
            matches, current_fifa, fifa_history_for_run
        )
        print(f"  {len(elos)} teams rated")

        print("  Estimating Dixon-Coles rho...")
        dixon_coles_rho = estimate_dixon_coles_rho(dc_samples)
        print(f"  rho = {dixon_coles_rho}")

        print("  Training ML candidates + stacking...")
        ml_meta = train_ml_models(ml_rows)

        fifa_blend, elo_blend, ml_blend, decision_calibration = tune_blend_weights(
            ml_rows, ml_meta, dixon_coles_rho
        )
        system_metrics = evaluate_system_accuracy(
            ml_rows,
            ml_meta,
            dixon_coles_rho,
            fifa_blend,
            elo_blend,
            ml_blend,
            decision_calibration,
        )
        score = system_metrics.get("calibratedAccuracy", 0.0) if system_metrics else 0.0
        experiment_results.append(
            {
                **params,
                "selectedModel": ml_meta.get("type") if ml_meta else None,
                "fifaBlendWeight": fifa_blend,
                "eloBlendWeight": elo_blend,
                "mlBlendWeight": ml_blend,
                "decisionCalibration": decision_calibration,
                "systemMetrics": system_metrics,
            }
        )
        if system_metrics:
            print(
                f"  Holdout accuracy - Poisson: {system_metrics['poissonAccuracy']:.3f}, "
                f"ML: {system_metrics.get('mlAccuracyHoldout', 0):.3f}, "
                f"Combined: {system_metrics['combinedAccuracy']:.3f}, "
                f"Calibrated: {system_metrics['calibratedAccuracy']:.3f}"
            )

        if best_run is None or score > best_run["score"]:
            best_run = {
                "score": score,
                "params": params,
                "elos": elos,
                "live_state": live_state,
                "dixon_coles_rho": dixon_coles_rho,
                "ml_meta": ml_meta,
                "fifa_blend": fifa_blend,
                "elo_blend": elo_blend,
                "ml_blend": ml_blend,
                "decision_calibration": decision_calibration,
                "system_metrics": system_metrics,
            }
    if best_run is None:
        raise RuntimeError("No model experiment completed")

    apply_model_params(best_run["params"])
    elos = best_run["elos"]
    live_state = best_run["live_state"]
    dixon_coles_rho = best_run["dixon_coles_rho"]
    ml_meta = best_run["ml_meta"]
    fifa_blend = best_run["fifa_blend"]
    elo_blend = best_run["elo_blend"]
    ml_blend = best_run["ml_blend"]
    decision_calibration = best_run["decision_calibration"]
    system_metrics = best_run["system_metrics"]

    print(
        f"\nSelected experiment: {best_run['params']['name']} "
        f"(combined holdout {best_run['score']:.3f})"
    )

    print("Building scorer stats...")
    scorers = load_scorers()
    scorer_stats = build_scorer_stats(scorers)
    print(f"  {len(scorer_stats)} teams with scorer data")

    teams_sorted = sorted(elos.keys())

    model_config = {
        "generatedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "matchCount": len(matches),
        "teamCount": len(elos),
        "homeAdvantageGoals": HOME_ADV_GOALS,
        "fifaBlendWeight": fifa_blend,
        "featureFifaBlendWeight": FIFA_BLEND_WEIGHT,
        "eloBlendWeight": elo_blend,
        "mlBlendWeight": ml_blend,
        "decisionCalibration": decision_calibration,
        "maxGoals": MAX_GOALS,
        "dixonColesRho": dixon_coles_rho,
        "scorerHalfLifeDays": SCORER_HALF_LIFE_DAYS,
        "selectedExperiment": best_run["params"],
        "experimentResults": experiment_results,
        "systemMetrics": system_metrics,
        "ml": ml_meta,
    }

    with open(OUT_DIR / "teams.json", "w", encoding="utf-8") as f:
        json.dump({"teams": elos, "teamList": teams_sorted}, f, indent=2)

    with open(OUT_DIR / "scorers.json", "w", encoding="utf-8") as f:
        json.dump({"scorers": scorer_stats}, f, indent=2)

    with open(OUT_DIR / "liveState.json", "w", encoding="utf-8") as f:
        json.dump(live_state, f, indent=2)

    with open(OUT_DIR / "model.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    print(f"\nDone! Output written to {OUT_DIR}")


if __name__ == "__main__":
    main()
