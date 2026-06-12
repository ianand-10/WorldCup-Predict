import teamsData from "../../public/data/teams.json";
import scorersData from "../../public/data/scorers.json";
import modelConfig from "../../public/data/model.json";
import liveStateData from "../../public/data/liveState.json";

import type {
  BinaryLogisticModel,
  GoalscorerPrediction,
  LiveState,
  ModelConfig,
  PredictionResult,
  ScorersData,
  TeamRatings,
  TeamsData,
  VenueType,
} from "./types";
import { outcomeProbabilities, scorelineMatrix } from "./poisson";

const teams = teamsData as TeamsData;
const scorers = scorersData as ScorersData;
const config = modelConfig as ModelConfig;
const liveState = liveStateData as LiveState;

const MAX_REST_DAYS = 90;
const DEFAULT_FIFA_POINTS = 1500;

export function getTeamList(): string[] {
  return teams.teamList;
}

export function getTeamRatings(team: string): TeamRatings | null {
  return teams.teams[team] ?? null;
}

function blendWithFifa(eloRating: number, fifaPts: number, weight: number): number {
  return (1 - weight) * eloRating + weight * fifaPts;
}

function blendedRatings(
  ratings: TeamRatings,
  team: string
): { offense: number; defense: number } {
  const fifa = fifaPoints(team);
  const weight = config.fifaBlendWeight ?? 0.3;
  return {
    offense: blendWithFifa(ratings.offense, fifa, weight),
    defense: blendWithFifa(ratings.defense, fifa, weight),
  };
}

function goalExpectation(
  offRating: number,
  defRating: number,
  venueBoost: number = 0
): number {
  const diff = (offRating - defRating) / 400;
  const base = Math.exp(0.55 * diff + venueBoost - 0.15);
  return Math.max(0.15, Math.min(4.5, base));
}

function pairKey(teamA: string, teamB: string): string {
  return [teamA, teamB].sort().join("|");
}

function normalizeRestDays(days: number): number {
  return Math.min(Math.max(days, 0), MAX_REST_DAYS) / MAX_REST_DAYS;
}

function daysSinceLastMatch(team: string): number {
  const ref = new Date(liveState.referenceDate);
  const last = liveState.lastMatchDate[team];
  if (!last) return 14;
  const lastDate = new Date(last);
  const diff = (ref.getTime() - lastDate.getTime()) / (1000 * 60 * 60 * 24);
  return Math.min(Math.max(diff, 0), MAX_REST_DAYS);
}

function h2hRateForTeam(teamA: string, teamB: string, forTeam: string): number {
  const key = pairKey(teamA, teamB);
  const entry = liveState.h2h[key];
  if (!entry) return 0.5;
  const rate = entry[forTeam];
  return typeof rate === "number" ? rate : 0.5;
}

function fifaPoints(team: string): number {
  return liveState.fifaPoints[team] ?? DEFAULT_FIFA_POINTS;
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function buildFeatureVector(
  homeTeam: string,
  awayTeam: string,
  isNeutral: boolean,
  ratingsHome: TeamRatings,
  ratingsAway: TeamRatings
): number[] {
  const homeDays = daysSinceLastMatch(homeTeam);
  const awayDays = daysSinceLastMatch(awayTeam);

  const venueBoost = isNeutral ? 0 : config.homeAdvantageGoals;
  const expHome = goalExpectation(
    ratingsHome.offense,
    ratingsAway.defense,
    isNeutral ? 0 : venueBoost + ratingsHome.homeBonus
  );
  const expAway = goalExpectation(
    ratingsAway.offense,
    ratingsHome.defense,
    ratingsAway.awayPenalty
  );

  return [
    ratingsHome.overall - ratingsAway.overall,
    ratingsHome.offense - ratingsAway.offense,
    ratingsHome.defense - ratingsAway.defense,
    Math.abs(ratingsHome.overall - ratingsAway.overall),
    isNeutral ? 0 : 1,
    isNeutral ? 0 : ratingsHome.homeBonus - ratingsAway.awayPenalty,
    liveState.form[homeTeam] ?? 0.5,
    liveState.form[awayTeam] ?? 0.5,
    h2hRateForTeam(homeTeam, awayTeam, homeTeam),
    normalizeRestDays(homeDays),
    normalizeRestDays(awayDays),
    (homeDays - awayDays) / MAX_REST_DAYS,
    fifaPoints(homeTeam) - fifaPoints(awayTeam),
    expHome - expAway,
    Math.abs(expHome - expAway) < 0.45 ? 1 : 0,
  ];
}

function scaleFeatures(raw: number[]): number[] {
  const ml = config.ml;
  if (!ml) return raw;
  return raw.map(
    (f, i) => (f - ml.scalerMean[i]) / (ml.scalerScale[i] || 1)
  );
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function binaryLogisticProbability(
  model: BinaryLogisticModel,
  features: number[]
): number {
  const logit =
    model.intercept +
    model.coefficients.reduce((sum, coef, i) => sum + coef * features[i], 0);
  return sigmoid(logit);
}

function mlPredictFromHomePerspective(
  scaled: number[]
): { winHome: number; draw: number; winAway: number } | null {
  const ml = config.ml;
  if (!ml?.outcomeModel || !ml?.drawModel) return null;

  const logits = ml.outcomeModel.intercepts.map((intercept, c) =>
    intercept +
    ml.outcomeModel.coefficients[c].reduce(
      (sum, coef, i) => sum + coef * scaled[i],
      0
    )
  );
  const base = softmax(logits);

  const drawBlend = ml.drawBlendWeight ?? 0.35;
  const pDrawSpecialist = binaryLogisticProbability(ml.drawModel, scaled);
  const draw = (1 - drawBlend) * base[1] + drawBlend * pDrawSpecialist;
  const total = base[0] + draw + base[2];

  return {
    winHome: base[0] / total,
    draw: draw / total,
    winAway: base[2] / total,
  };
}

function mlPredict(
  teamA: string,
  teamB: string,
  venue: VenueType,
  ratingsA: TeamRatings,
  ratingsB: TeamRatings
): { winA: number; draw: number; winB: number } | null {
  if (!config.ml) return null;

  if (venue === "teamA_home") {
    const raw = buildFeatureVector(teamA, teamB, false, ratingsA, ratingsB);
    const scaled = scaleFeatures(raw);
    const p = mlPredictFromHomePerspective(scaled);
    if (!p) return null;
    return { winA: p.winHome, draw: p.draw, winB: p.winAway };
  }

  if (venue === "teamB_home") {
    const raw = buildFeatureVector(teamB, teamA, false, ratingsB, ratingsA);
    const scaled = scaleFeatures(raw);
    const p = mlPredictFromHomePerspective(scaled);
    if (!p) return null;
    return { winA: p.winAway, draw: p.draw, winB: p.winHome };
  }

  const rawA = buildFeatureVector(teamA, teamB, true, ratingsA, ratingsB);
  const rawB = buildFeatureVector(teamB, teamA, true, ratingsB, ratingsA);
  const scaledA = scaleFeatures(rawA);
  const scaledB = scaleFeatures(rawB);
  const pA = mlPredictFromHomePerspective(scaledA);
  const pB = mlPredictFromHomePerspective(scaledB);
  if (!pA || !pB) return null;

  const winA = (pA.winHome + pB.winAway) / 2;
  const winB = (pA.winAway + pB.winHome) / 2;
  const draw = (pA.draw + pB.draw) / 2;
  const total = winA + draw + winB;

  return {
    winA: winA / total,
    draw: draw / total,
    winB: winB / total,
  };
}

function predictScorers(
  team: string,
  expectedGoals: number
): GoalscorerPrediction[] {
  const players = scorers.scorers[team];
  if (!players || players.length === 0) return [];

  return players.slice(0, 6).map((p) => ({
    name: p.name,
    goals: p.goals,
    probability: Math.min(0.85, expectedGoals * p.share),
  }));
}

export function predictMatch(
  teamA: string,
  teamB: string,
  venue: VenueType
): PredictionResult | null {
  const ratingsA = getTeamRatings(teamA);
  const ratingsB = getTeamRatings(teamB);
  if (!ratingsA || !ratingsB) return null;

  const blendedA = blendedRatings(ratingsA, teamA);
  const blendedB = blendedRatings(ratingsB, teamB);

  let venueBoostA = 0;
  let venueBoostB = 0;

  if (venue === "teamA_home") {
    venueBoostA = config.homeAdvantageGoals + ratingsA.homeBonus;
    venueBoostB = ratingsB.awayPenalty;
  } else if (venue === "teamB_home") {
    venueBoostB = config.homeAdvantageGoals + ratingsB.homeBonus;
    venueBoostA = ratingsA.awayPenalty;
  }

  const lambdaA = goalExpectation(blendedA.offense, blendedB.defense, venueBoostA);
  const lambdaB = goalExpectation(blendedB.offense, blendedA.defense, venueBoostB);

  const [homeLambda, awayLambda] =
    venue === "teamB_home" ? [lambdaB, lambdaA] : [lambdaA, lambdaB];

  const rho = config.dixonColesRho ?? 0;
  const matrix = scorelineMatrix(homeLambda, awayLambda, config.maxGoals, rho);

  const mappedScorelines = matrix.map((s) => ({
    home: venue === "teamB_home" ? s.away : s.home,
    away: venue === "teamB_home" ? s.home : s.away,
    probability: s.probability,
  }));

  const eloOutcomes = outcomeProbabilities(matrix, venue !== "teamB_home");
  const mlOutcomes = mlPredict(teamA, teamB, venue, ratingsA, ratingsB);

  let winA = eloOutcomes.winA;
  let draw = eloOutcomes.draw;
  let winB = eloOutcomes.winB;

  if (mlOutcomes) {
    const ew = config.eloBlendWeight;
    const mw = config.mlBlendWeight;
    winA = ew * eloOutcomes.winA + mw * mlOutcomes.winA;
    draw = ew * eloOutcomes.draw + mw * mlOutcomes.draw;
    winB = ew * eloOutcomes.winB + mw * mlOutcomes.winB;
    const total = winA + draw + winB;
    winA /= total;
    draw /= total;
    winB /= total;
  }

  return {
    teamA,
    teamB,
    venue,
    expectedGoalsA: Math.round(lambdaA * 100) / 100,
    expectedGoalsB: Math.round(lambdaB * 100) / 100,
    winProbabilityA: winA,
    drawProbability: draw,
    winProbabilityB: winB,
    topScorelines: mappedScorelines.slice(0, 15),
    scorersA: predictScorers(teamA, lambdaA),
    scorersB: predictScorers(teamB, lambdaB),
    ratingsA,
    ratingsB,
  };
}

export function getModelMeta() {
  const sys = config.systemMetrics;
  return {
    matchCount: config.matchCount,
    teamCount: config.teamCount,
    generatedAt: config.generatedAt,
    mlAccuracy: config.ml?.accuracy ?? null,
    mlAccuracyStd: config.ml?.accuracyStd ?? null,
    combinedAccuracy: sys?.combinedAccuracy ?? null,
    poissonAccuracy: sys?.poissonAccuracy ?? null,
    mlHoldoutAccuracy: sys?.mlAccuracyHoldout ?? config.ml?.holdoutAccuracy ?? null,
    fifaBlendWeight: config.fifaBlendWeight ?? null,
    eloBlendWeight: config.eloBlendWeight,
    mlBlendWeight: config.mlBlendWeight,
    dixonColesRho: config.dixonColesRho ?? null,
  };
}
