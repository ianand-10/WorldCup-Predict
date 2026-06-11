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
}

export interface ScorersData {
  scorers: Record<string, ScorerPlayer[]>;
}

export interface MLModel {
  type: string;
  featureNames: string[];
  classes: string[];
  accuracy: number;
  scalerMean: number[];
  scalerScale: number[];
  coefficients: number[][];
  intercepts: number[];
}

export interface ModelConfig {
  generatedAt: string;
  matchCount: number;
  teamCount: number;
  homeAdvantageGoals: number;
  eloBlendWeight: number;
  mlBlendWeight: number;
  maxGoals: number;
  ml: MLModel | null;
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
