export interface BinaryLogisticModel {
  coefficients: number[];
  intercept: number;
}

export interface TeamRatings {
  overall: number;
  offense: number;
  defense: number;
  homeBonus: number;
  awayPenalty: number;
}

export interface TeamsData {
  teams: Record<string, TeamRatings>;
  teamList: string[];
}

export interface ScorerPlayer {
  name: string;
  goals: number;
  share: number;
  weightedGoals?: number;
}

export interface ScorersData {
  scorers: Record<string, ScorerPlayer[]>;
}

export interface MLModel {
  type: string;
  featureNames: string[];
  classes: string[];
  accuracy: number;
  accuracyStd?: number;
  holdoutAccuracy?: number;
  cvFolds?: number;
  drawBlendWeight?: number;
  scalerMean: number[];
  scalerScale: number[];
  outcomeModel: {
    coefficients: number[][];
    intercepts: number[];
  };
  drawModel: BinaryLogisticModel;
}

export interface SystemMetrics {
  poissonAccuracy: number;
  mlAccuracyHoldout?: number | null;
  combinedAccuracy: number;
  holdoutSize: number;
}

export interface ModelConfig {
  generatedAt: string;
  matchCount: number;
  teamCount: number;
  homeAdvantageGoals: number;
  fifaBlendWeight?: number;
  eloBlendWeight: number;
  mlBlendWeight: number;
  maxGoals: number;
  dixonColesRho?: number;
  scorerHalfLifeDays?: number;
  systemMetrics?: SystemMetrics;
  ml: MLModel | null;
}

export interface LiveState {
  referenceDate: string;
  form: Record<string, number>;
  lastMatchDate: Record<string, string>;
  h2h: Record<string, Record<string, number | string>>;
  fifaPoints: Record<string, number>;
}

export type VenueType = "teamA_home" | "neutral" | "teamB_home";

export interface Scoreline {
  home: number;
  away: number;
  probability: number;
}

export interface GoalscorerPrediction {
  name: string;
  probability: number;
  goals: number;
}

export interface PredictionResult {
  teamA: string;
  teamB: string;
  venue: VenueType;
  expectedGoalsA: number;
  expectedGoalsB: number;
  winProbabilityA: number;
  drawProbability: number;
  winProbabilityB: number;
  topScorelines: Scoreline[];
  scorersA: GoalscorerPrediction[];
  scorersB: GoalscorerPrediction[];
  ratingsA: TeamRatings;
  ratingsB: TeamRatings;
}
