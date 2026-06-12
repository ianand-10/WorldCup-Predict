"use client";

import { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

import AnimatedBackground from "./AnimatedBackground";
import TeamSelector from "./TeamSelector";
import VenueSelector from "./VenueSelector";
import ProbabilityOrb from "./ProbabilityOrb";
import ScorelineCascade from "./ScorelineCascade";
import EloRadar from "./EloRadar";
import GoalscorerPills from "./GoalscorerPills";

import { getModelMeta, getTeamList, predictMatch } from "@/lib/predict";
import type { PredictionResult, VenueType } from "@/lib/types";

export default function MatchPredictor() {
  const teams = useMemo(() => getTeamList(), []);
  const meta = useMemo(() => getModelMeta(), []);

  const [teamA, setTeamA] = useState("");
  const [teamB, setTeamB] = useState("");
  const [venue, setVenue] = useState<VenueType>("neutral");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [hasPredicted, setHasPredicted] = useState(false);
  const [showMethodology, setShowMethodology] = useState(false);

  const canPredict = teamA && teamB && teamA !== teamB;

  function handlePredict() {
    if (!canPredict) return;
    const prediction = predictMatch(teamA, teamB, venue);
    setResult(prediction);
    setHasPredicted(true);
  }

  return (
    <>
      <AnimatedBackground venue={venue} />

      <div className="relative mx-auto min-h-screen max-w-7xl px-4 py-8 sm:px-6 lg:px-8 lg:py-12">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-10 lg:mb-14"
        >
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.3em] text-accent-emerald">
                International Football Intelligence
              </p>
              <h1 className="text-3xl font-bold tracking-tight text-white sm:text-4xl lg:text-5xl">
                World Cup{" "}
                <span className="text-gradient-gold">Predictor</span>
              </h1>
              <p className="mt-3 max-w-lg text-sm leading-relaxed text-muted-light">
                Multi-dimensional ELO ratings, Poisson scorelines, and ML-calibrated
                outcomes — trained on {meta.matchCount.toLocaleString()} international
                matches since 2021.
              </p>
            </div>
            <div className="glass-panel rounded-xl px-4 py-3 text-right">
              <p className="font-mono text-[10px] uppercase tracking-wider text-muted">
                Model Data
              </p>
              <p className="font-mono text-sm text-white">
                {meta.teamCount} teams
              </p>
              {meta.combinedAccuracy != null && (
                <p className="font-mono text-[10px] text-accent-gold">
                  Combined accuracy {(meta.combinedAccuracy * 100).toFixed(1)}%
                </p>
              )}
              {meta.mlAccuracy != null && (
                <p className="font-mono text-[10px] text-accent-emerald">
                  ML CV {(meta.mlAccuracy * 100).toFixed(1)}%
                  {meta.mlAccuracyStd != null &&
                    ` ± ${(meta.mlAccuracyStd * 100).toFixed(1)}%`}
                </p>
              )}
              {meta.poissonAccuracy != null && (
                <p className="font-mono text-[10px] text-muted">
                  Poisson {(meta.poissonAccuracy * 100).toFixed(1)}%
                </p>
              )}
            </div>
          </div>
        </motion.header>

        <div className="grid gap-8 lg:grid-cols-[380px_1fr] lg:gap-12">
          {/* Input Panel */}
          <motion.aside
            initial={{ opacity: 0, x: -24 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="space-y-6"
          >
            <div className="glass-panel rounded-2xl p-6 space-y-5">
              <TeamSelector
                label="Team A"
                teams={teams}
                value={teamA}
                onChange={setTeamA}
                accent="emerald"
                exclude={teamB}
              />
              <TeamSelector
                label="Team B"
                teams={teams}
                value={teamB}
                onChange={setTeamB}
                accent="gold"
                exclude={teamA}
              />
              <VenueSelector
                venue={venue}
                onChange={setVenue}
                teamA={teamA}
                teamB={teamB}
              />

              <motion.button
                type="button"
                onClick={handlePredict}
                disabled={!canPredict}
                whileHover={canPredict ? { scale: 1.02 } : {}}
                whileTap={canPredict ? { scale: 0.98 } : {}}
                className={`relative w-full overflow-hidden rounded-xl py-4 text-sm font-semibold uppercase tracking-widest transition-all duration-300 ${
                  canPredict
                    ? "bg-gradient-to-r from-accent-emerald to-accent-emerald/80 text-white glow-emerald hover:shadow-lg"
                    : "cursor-not-allowed bg-white/5 text-muted"
                }`}
              >
                {canPredict ? "Generate Prediction" : "Select Two Teams"}
                {canPredict && (
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"
                    animate={{ x: ["-100%", "200%"] }}
                    transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
                  />
                )}
              </motion.button>
            </div>

            {/* Decorative stat strip */}
            <div className="hidden lg:flex items-center gap-3 px-2">
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-white/10 to-transparent" />
              <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
                ELO · Poisson · ML · FIFA
              </span>
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-white/10 to-transparent" />
            </div>

            <div className="glass-panel rounded-2xl p-5">
              <button
                type="button"
                onClick={() => setShowMethodology((v) => !v)}
                className="flex w-full items-center justify-between text-left"
              >
                <span className="text-xs font-medium uppercase tracking-widest text-muted-light">
                  How predictions work
                </span>
                <span className="font-mono text-xs text-accent-emerald">
                  {showMethodology ? "−" : "+"}
                </span>
              </button>
              <AnimatePresence>
                {showMethodology && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="mt-4 space-y-3 text-xs leading-relaxed text-muted-light">
                      <p>
                        <strong className="text-white">1. Team strength (ELO + FIFA).</strong>{" "}
                        Each team has offense, defense, and overall ELO ratings built from
                        every international match since 2021, weighted by tournament
                        importance. These are blended with current FIFA ranking points so
                        teams that only beat weaker opponents don&apos;t get overstated.
                      </p>
                      <p>
                        <strong className="text-white">2. Expected goals (Poisson).</strong>{" "}
                        Offense vs. defense ratings produce expected goals (λ) for each
                        side, with home advantage and venue bonuses applied. A Dixon-Coles
                        adjustment corrects low-scoring correlation (0-0, 1-1 draws).
                      </p>
                      <p>
                        <strong className="text-white">3. Scorelines.</strong> All score
                        combinations up to {8} goals are computed from independent Poisson
                        distributions, adjusted by ρ={meta.dixonColesRho ?? "−0.01"}.
                        Probabilities are normalized; the most likely scorelines are shown
                        in the cascade.
                      </p>
                      <p>
                        <strong className="text-white">4. Win / draw / loss.</strong>{" "}
                        Poisson scoreline probabilities sum into outcome odds. A separate
                        ML model (logistic regression on ELO diff, form, head-to-head, rest
                        days, FIFA gap, and expected-goal diff) calibrates draws and upsets.
                        Final odds blend Poisson ({((meta.eloBlendWeight ?? 0.55) * 100).toFixed(0)}%)
                        and ML ({((meta.mlBlendWeight ?? 0.45) * 100).toFixed(0)}%) on
                        chronological holdout data.
                      </p>
                      <p>
                        <strong className="text-white">5. Likely scorers.</strong>{" "}
                        Player probabilities use recency-weighted goal shares (12-month
                        half-life), scaled by the team&apos;s expected goals in this match.
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.aside>

          {/* Results Panel */}
          <main>
            <AnimatePresence mode="wait">
              {hasPredicted && result ? (
                <motion.div
                  key={`${teamA}-${teamB}-${venue}`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.5 }}
                  className="space-y-6"
                >
                  <div className="glass-panel rounded-2xl p-6 lg:p-8">
                    <ProbabilityOrb
                      teamA={result.teamA}
                      teamB={result.teamB}
                      winA={result.winProbabilityA}
                      draw={result.drawProbability}
                      winB={result.winProbabilityB}
                      expectedA={result.expectedGoalsA}
                      expectedB={result.expectedGoalsB}
                    />
                  </div>

                  <EloRadar
                    teamA={result.teamA}
                    teamB={result.teamB}
                    ratingsA={result.ratingsA}
                    ratingsB={result.ratingsB}
                  />

                  <div className="glass-panel rounded-2xl p-6 lg:p-8">
                    <ScorelineCascade
                      scorelines={result.topScorelines}
                      teamA={result.teamA}
                      teamB={result.teamB}
                    />
                  </div>

                  <div className="grid gap-6 md:grid-cols-2">
                    <div className="glass-panel rounded-2xl p-6">
                      <GoalscorerPills
                        team={result.teamA}
                        scorers={result.scorersA}
                        accent="emerald"
                      />
                    </div>
                    <div className="glass-panel rounded-2xl p-6">
                      <GoalscorerPills
                        team={result.teamB}
                        scorers={result.scorersB}
                        accent="gold"
                      />
                    </div>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex min-h-[400px] flex-col items-center justify-center glass-panel rounded-2xl p-8 text-center"
                >
                  <div className="mb-6 h-24 w-24 rounded-full border border-white/10 flex items-center justify-center">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                      className="h-20 w-20 rounded-full border border-dashed border-accent-emerald/30"
                    />
                  </div>
                  <h2 className="text-lg font-semibold text-white">
                    Awaiting Match Selection
                  </h2>
                  <p className="mt-2 max-w-sm text-sm text-muted-light">
                    Choose two international teams and a venue to generate
                    scoreline probabilities, win odds, and likely goalscorers.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </main>
        </div>

        <footer className="mt-16 border-t border-white/5 pt-8 text-center">
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted">
            Powered by ELO ratings · Poisson distribution · Logistic regression ML
          </p>
        </footer>
      </div>
    </>
  );
}
