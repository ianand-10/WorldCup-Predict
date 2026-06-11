import teamsData from "../../public/data/teams.json";
import scorersData from "../../public/data/scorers.json";
import modelConfig from "../../public/data/model.json";

import type {
  GoalscorerPrediction,
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

export function getTeamList(): string[] {
  return teams.teamList;
}

export function getTeamRatings(team: string): TeamRatings | null {
  return teams.teams[team] ?? null;
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

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function mlPredict(
  teamA: string,
  teamB: string,
  venue: VenueType,
  ratingsA: TeamRatings,
  ratingsB: TeamRatings
): { winA: number; draw: number; winB: number } | null {
  const ml = config.ml;
  if (!ml) return null;

  const features = [
    ratingsA.overall - ratingsB.overall,
    ratingsA.offense - ratingsB.offense,
    ratingsA.defense - ratingsB.defense,
    venue === "neutral" ? 0 : venue === "teamA_home" ? 1 : 0,
    venue === "teamA_home"
      ? ratingsA.homeBonus - ratingsB.awayPenalty
      : venue === "teamB_home"
        ? ratingsB.homeBonus - ratingsA.awayPenalty
        : 0,
    0.5,
    0.5,
    0.5,
  ];

  const scaled = features.map(
    (f, i) => (f - ml.scalerMean[i]) / ml.scalerScale[i]
  );

  const logits = ml.intercepts.map((intercept, c) =>
    intercept + ml.coefficients[c].reduce((s, coef, i) => s + coef * scaled[i], 0)
  );

  const probs = softmax(logits);

  if (venue === "teamA_home") {
    return { winA: probs[0], draw: probs[1], winB: probs[2] };
  }
  if (venue === "teamB_home") {
    return { winA: probs[2], draw: probs[1], winB: probs[0] };
  }

  const eloDiff = ratingsA.overall - ratingsB.overall;
  const shift = eloDiff / 800;
  return {
    winA: probs[0] * 0.5 + probs[2] * 0.5 + shift,
    draw: probs[1],
    winB: probs[2] * 0.5 + probs[0] * 0.5 - shift,
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

  let venueBoostA = 0;
  let venueBoostB = 0;

  if (venue === "teamA_home") {
    venueBoostA = config.homeAdvantageGoals + ratingsA.homeBonus;
    venueBoostB = ratingsB.awayPenalty;
  } else if (venue === "teamB_home") {
    venueBoostB = config.homeAdvantageGoals + ratingsB.homeBonus;
    venueBoostA = ratingsA.awayPenalty;
  }

  const lambdaA = goalExpectation(ratingsA.offense, ratingsB.defense, venueBoostA);
  const lambdaB = goalExpectation(ratingsB.offense, ratingsA.defense, venueBoostB);

  const [homeLambda, awayLambda] =
    venue === "teamB_home" ? [lambdaB, lambdaA] : [lambdaA, lambdaB];

  const matrix = scorelineMatrix(homeLambda, awayLambda, config.maxGoals);

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
  return {
    matchCount: config.matchCount,
    teamCount: config.teamCount,
    generatedAt: config.generatedAt,
    mlAccuracy: config.ml?.accuracy ?? null,
  };
}
