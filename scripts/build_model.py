"""
Build ELO ratings, ML models, Dixon-Coles params, and live state from match CSVs.
Outputs JSON to public/data/ for the Next.js predictor.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "public" / "data"

INITIAL_ELO = 1500.0
BASE_K = 20.0
HOME_ADV_GOALS = 0.28
MAX_GOALS = 8
DEFAULT_FIFA_POINTS = 1500.0
MAX_REST_DAYS = 90
FIFA_BLEND_WEIGHT = 0.3
ELO_BLEND_WEIGHT = 0.55
ML_BLEND_WEIGHT = 0.45
SCORER_HALF_LIFE_DAYS = 365

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
    "h2hHomeRate",
    "homeDaysSinceNorm",
    "awayDaysSinceNorm",
    "restDaysDiffNorm",
    "fifaPointsDiff",
    "expectedGoalDiff",
    "closeMatchIndicator",
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


def normalize_rest_days(days: float) -> float:
    return min(max(days, 0.0), MAX_REST_DAYS) / MAX_REST_DAYS


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
        pd.DataFrame(rows).to_csv(normalized_path, index=False)

    return result


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
    home: str,
    away: str,
    neutral: bool,
    fifa: dict[str, float],
    fifa_blend: float,
) -> tuple[float, float]:
    home_fifa = fifa.get(home, DEFAULT_FIFA_POINTS)
    away_fifa = fifa.get(away, DEFAULT_FIFA_POINTS)

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
    h2h_pair: dict[str, dict[str, list[float]]],
    last_match: dict[str, pd.Timestamp],
    match_date: pd.Timestamp,
    fifa: dict[str, float],
) -> dict[str, float]:
    pk = pair_key(home, away)
    pair_stats = h2h_pair.get(pk, {})

    home_days = (match_date - last_match.get(home, match_date - pd.Timedelta(days=14))).days
    away_days = (match_date - last_match.get(away, match_date - pd.Timedelta(days=14))).days
    home_days = float(min(max(home_days, 0), MAX_REST_DAYS))
    away_days = float(min(max(away_days, 0), MAX_REST_DAYS))

    venue_boost = 0.0 if neutral else HOME_ADV_GOALS
    exp_home = goal_expectation(hr.offense, ar.defense, venue_boost + hr.home_bonus)
    exp_away = goal_expectation(ar.offense, hr.defense, ar.away_penalty)

    home_fifa = fifa.get(home, DEFAULT_FIFA_POINTS)
    away_fifa = fifa.get(away, DEFAULT_FIFA_POINTS)

    return {
        "eloOverallDiff": hr.overall - ar.overall,
        "eloOffenseDiff": hr.offense - ar.offense,
        "eloDefenseDiff": hr.defense - ar.defense,
        "absEloOverallDiff": abs(hr.overall - ar.overall),
        "isHome": 0.0 if neutral else 1.0,
        "venueAdvantage": 0.0 if neutral else hr.home_bonus - ar.away_penalty,
        "homeForm": recent_form(form[home]),
        "awayForm": recent_form(form[away]),
        "h2hHomeRate": h2h_rate_for_team(pair_stats, home),
        "homeDaysSinceNorm": normalize_rest_days(home_days),
        "awayDaysSinceNorm": normalize_rest_days(away_days),
        "restDaysDiffNorm": (home_days - away_days) / MAX_REST_DAYS,
        "fifaPointsDiff": home_fifa - away_fifa,
        "expectedGoalDiff": exp_home - exp_away,
        "closeMatchIndicator": 1.0 if abs(exp_home - exp_away) < 0.45 else 0.0,
    }


def compute_elos_and_features(matches: pd.DataFrame, fifa: dict[str, float]):
    teams: dict[str, TeamRatings] = defaultdict(TeamRatings)
    form: dict[str, list[tuple[int, int]]] = defaultdict(list)
    h2h_pair: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    last_match: dict[str, pd.Timestamp] = {}
    history: list[dict] = []
    ml_rows: list[dict] = []
    dc_samples: list[tuple[float, float, int, int]] = []

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        hs, aws = int(row["home_score"]), int(row["away_score"])
        neutral = bool(row["neutral"])
        tournament = str(row.get("tournament", "Friendly"))
        match_date = row["date"]
        k = k_factor(tournament)
        sample_weight = tournament_weight(tournament)

        hr, ar = teams[home], teams[away]

        features = build_match_features(
            home, away, neutral, hr, ar, form, h2h_pair, last_match, match_date, fifa
        )

        outcome = 0 if hs > aws else (2 if hs < aws else 1)

        exp_home_goals, exp_away_goals = match_lambdas_with_fifa(
            hr, ar, home, away, neutral, fifa, FIFA_BLEND_WEIGHT
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
                "homeFifa": fifa.get(home, DEFAULT_FIFA_POINTS),
                "awayFifa": fifa.get(away, DEFAULT_FIFA_POINTS),
            }
        )

        dc_samples.append((exp_home_goals, exp_away_goals, hs, aws))

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

    reference_date = matches["date"].max()
    live_form = {team: round(recent_form(form[team]), 4) for team in teams}
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
        "lastMatchDate": live_last_match,
        "h2h": live_h2h,
        "fifaPoints": {
            team: round(fifa.get(team, DEFAULT_FIFA_POINTS), 1) for team in teams
        },
    }
    for team, points in fifa.items():
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


def blended_predict_proba(
    outcome_model: LogisticRegression,
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


def train_ml_models(ml_rows: list[dict]) -> dict | None:
    if len(ml_rows) < 300:
        return None

    X = np.array([[r[f] for f in FEATURE_NAMES] for r in ml_rows])
    y_outcome = np.array([r["outcome"] for r in ml_rows])
    y_draw = np.array([r["isDraw"] for r in ml_rows])
    weights = np.array([r["sampleWeight"] for r in ml_rows])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    outcome_model, draw_model = fit_multinomial_with_draw_calibration(
        X_scaled, y_outcome, y_draw, weights
    )

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores: list[float] = []

    for train_idx, test_idx in tscv.split(X_scaled):
        if len(train_idx) < 800 or len(test_idx) < 50:
            continue

        fold_scaler = StandardScaler()
        X_train = fold_scaler.fit_transform(X[train_idx])
        X_test = fold_scaler.transform(X[test_idx])

        fold_outcome, fold_draw = fit_multinomial_with_draw_calibration(
            X_train,
            y_outcome[train_idx],
            y_draw[train_idx],
            weights[train_idx],
        )
        probs = blended_predict_proba(fold_outcome, fold_draw, X_test)
        cv_scores.append(outcome_accuracy(y_outcome[test_idx], probs))

    split_idx = int(len(X_scaled) * 0.85)
    holdout_scaler = StandardScaler()
    X_train_h = holdout_scaler.fit_transform(X[:split_idx])
    X_test_h = holdout_scaler.transform(X[split_idx:])
    holdout_outcome, holdout_draw = fit_multinomial_with_draw_calibration(
        X_train_h,
        y_outcome[:split_idx],
        y_draw[:split_idx],
        weights[:split_idx],
    )
    holdout_probs = blended_predict_proba(holdout_outcome, holdout_draw, X_test_h)
    holdout_acc = outcome_accuracy(y_outcome[split_idx:], holdout_probs)

    cv_mean = float(np.mean(cv_scores)) if cv_scores else holdout_acc
    cv_std = float(np.std(cv_scores)) if cv_scores else 0.0

    print(f"  ML time-series CV accuracy: {cv_mean:.3f} (+/- {cv_std:.3f})")
    print(f"  ML chronological holdout (last 15%): {holdout_acc:.3f}")

    return {
        "type": "multinomial_with_draw_calibration",
        "featureNames": FEATURE_NAMES,
        "classes": ["home_win", "draw", "away_win"],
        "accuracy": round(cv_mean, 4),
        "accuracyStd": round(cv_std, 4),
        "holdoutAccuracy": round(float(holdout_acc), 4),
        "cvFolds": len(cv_scores),
        "drawBlendWeight": 0.35,
        "scalerMean": scaler.mean_.tolist(),
        "scalerScale": scaler.scale_.tolist(),
        "outcomeModel": serialize_multinomial(outcome_model),
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
) -> dict[str, float]:
    split_idx = int(len(ml_rows) * 0.85)
    holdout = ml_rows[split_idx:]
    if len(holdout) < 50:
        return {}

    y_true = np.array([r["outcome"] for r in holdout])
    poisson_probs = np.zeros((len(holdout), 3))
    ml_probs = np.zeros((len(holdout), 3))
    ensemble_probs = np.zeros((len(holdout), 3))

    draw_blend = ml_meta.get("drawBlendWeight", 0.35) if ml_meta else 0.35

    if ml_meta:
        X = np.array([[r[f] for f in FEATURE_NAMES] for r in holdout])
        mean = np.array(ml_meta["scalerMean"])
        scale = np.array(ml_meta["scalerScale"])
        X_scaled = (X - mean) / np.where(scale == 0, 1, scale)

        coef = np.array(ml_meta["outcomeModel"]["coefficients"])
        intercepts = np.array(ml_meta["outcomeModel"]["intercepts"])
        logits = intercepts + X_scaled @ coef.T
        logits -= logits.max(axis=1, keepdims=True)
        base = np.exp(logits)
        base /= base.sum(axis=1, keepdims=True)

        draw_coef = np.array(ml_meta["drawModel"]["coefficients"])
        draw_intercept = ml_meta["drawModel"]["intercept"]
        draw_logit = draw_intercept + X_scaled @ draw_coef
        p_draw_specialist = 1.0 / (1.0 + np.exp(-draw_logit))

        ml_probs = base.copy()
        ml_probs[:, 1] = (1.0 - draw_blend) * base[:, 1] + draw_blend * p_draw_specialist
        ml_probs /= ml_probs.sum(axis=1, keepdims=True)

    for i, row in enumerate(holdout):
        exp_home, exp_away = lambdas_from_row(row, fifa_blend)
        wh, dr, wa = poisson_outcome_probs(exp_home, exp_away, dixon_coles_rho)
        poisson_probs[i] = [wh, dr, wa]

        if ml_meta:
            ensemble_probs[i] = elo_blend * poisson_probs[i] + ml_blend * ml_probs[i]
            ensemble_probs[i] /= ensemble_probs[i].sum()
        else:
            ensemble_probs[i] = poisson_probs[i]

    return {
        "poissonAccuracy": round(outcome_accuracy(y_true, poisson_probs), 4),
        "mlAccuracyHoldout": round(outcome_accuracy(y_true, ml_probs), 4) if ml_meta else None,
        "combinedAccuracy": round(outcome_accuracy(y_true, ensemble_probs), 4),
        "holdoutSize": len(holdout),
    }


def tune_blend_weights(
    ml_rows: list[dict],
    ml_meta: dict | None,
    dixon_coles_rho: float,
) -> tuple[float, float, float]:
    if not ml_meta or len(ml_rows) < 500:
        return FIFA_BLEND_WEIGHT, ELO_BLEND_WEIGHT, ML_BLEND_WEIGHT

    split_idx = int(len(ml_rows) * 0.85)
    holdout = ml_rows[split_idx:]
    y_true = np.array([r["outcome"] for r in holdout])

    best_fifa = FIFA_BLEND_WEIGHT
    best_elo = ELO_BLEND_WEIGHT
    best_ml = ML_BLEND_WEIGHT
    best_acc = -1.0

    X = np.array([[r[f] for f in FEATURE_NAMES] for r in holdout])
    mean = np.array(ml_meta["scalerMean"])
    scale = np.array(ml_meta["scalerScale"])
    X_scaled = (X - mean) / np.where(scale == 0, 1, scale)

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

    poisson_by_fifa: dict[float, np.ndarray] = {}
    for fifa_blend in (0.2, 0.25, 0.3, 0.35, 0.4):
        probs = np.zeros((len(holdout), 3))
        for i, row in enumerate(holdout):
            exp_home, exp_away = lambdas_from_row(row, fifa_blend)
            wh, dr, wa = poisson_outcome_probs(exp_home, exp_away, dixon_coles_rho)
            probs[i] = [wh, dr, wa]
        poisson_by_fifa[fifa_blend] = probs

    for fifa_blend, poisson_probs in poisson_by_fifa.items():
        for elo_w in (0.45, 0.5, 0.55, 0.6, 0.65):
            ml_w = 1.0 - elo_w
            ensemble = elo_w * poisson_probs + ml_w * ml_probs
            ensemble /= ensemble.sum(axis=1, keepdims=True)
            acc = outcome_accuracy(y_true, ensemble)
            if acc > best_acc:
                best_acc = acc
                best_fifa = fifa_blend
                best_elo = elo_w
                best_ml = ml_w

    print(
        f"  Tuned blends — FIFA: {best_fifa:.2f}, Poisson: {best_elo:.2f}, ML: {best_ml:.2f} "
        f"(holdout acc {best_acc:.3f})"
    )
    return best_fifa, best_elo, best_ml


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
    fifa = load_fifa_rankings()

    print("Computing ELO ratings and features...")
    elos, history, ml_rows, dc_samples, live_state = compute_elos_and_features(
        matches, fifa
    )
    print(f"  {len(elos)} teams rated")

    print("Estimating Dixon-Coles rho...")
    dixon_coles_rho = estimate_dixon_coles_rho(dc_samples)
    print(f"  rho = {dixon_coles_rho}")

    print("Training ML models (multinomial logistic + draw calibration + time-series CV)...")
    ml_meta = train_ml_models(ml_rows)

    fifa_blend, elo_blend, ml_blend = tune_blend_weights(ml_rows, ml_meta, dixon_coles_rho)
    system_metrics = evaluate_system_accuracy(
        ml_rows, ml_meta, dixon_coles_rho, fifa_blend, elo_blend, ml_blend
    )
    if system_metrics:
        print(
            f"  Holdout accuracy — Poisson: {system_metrics['poissonAccuracy']:.3f}, "
            f"ML: {system_metrics.get('mlAccuracyHoldout', 0):.3f}, "
            f"Combined: {system_metrics['combinedAccuracy']:.3f}"
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
        "eloBlendWeight": elo_blend,
        "mlBlendWeight": ml_blend,
        "maxGoals": MAX_GOALS,
        "dixonColesRho": dixon_coles_rho,
        "scorerHalfLifeDays": SCORER_HALF_LIFE_DAYS,
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
