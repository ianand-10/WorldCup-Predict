"use client";

import { motion } from "framer-motion";
import type { Scoreline } from "@/lib/types";

interface Props {
  scorelines: Scoreline[];
  teamA: string;
  teamB: string;
}

export default function ScorelineCascade({ scorelines, teamA, teamB }: Props) {
  const maxProb = scorelines[0]?.probability ?? 1;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="space-y-3"
    >
      <div className="flex items-baseline justify-between">
        <h3 className="text-xs font-medium uppercase tracking-widest text-muted-light">
          Most Likely Scorelines
        </h3>
        <span className="font-mono text-[10px] text-muted">
          Poisson model
        </span>
      </div>

      <div className="space-y-2">
        {scorelines.map((s, i) => {
          const widthPct = (s.probability / maxProb) * 100;
          const isTop = i === 0;

          return (
            <motion.div
              key={`${s.home}-${s.away}`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.35, delay: i * 0.04 }}
              className="group relative"
            >
              <div className="mb-1 flex items-center justify-between text-sm">
                <span className={`font-mono font-medium ${isTop ? "text-accent-gold" : "text-white"}`}>
                  {s.home} – {s.away}
                </span>
                <span className="font-mono text-xs text-muted-light">
                  {(s.probability * 100).toFixed(1)}%
                </span>
              </div>

              <div className="relative h-2 overflow-hidden rounded-full bg-white/5">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${widthPct}%` }}
                  transition={{ duration: 0.6, delay: i * 0.04, ease: "easeOut" }}
                  className={`absolute inset-y-0 left-0 rounded-full ${
                    isTop
                      ? "animate-shimmer bg-gradient-to-r from-accent-gold/80 to-accent-emerald/80"
                      : "bg-gradient-to-r from-white/20 to-white/10"
                  }`}
                />
                {isTop && (
                  <div className="absolute inset-0 rounded-full animate-pulse-glow opacity-50" />
                )}
              </div>

              {isTop && (
                <p className="mt-1 text-[10px] text-muted">
                  {teamA} {s.home > s.away ? "wins" : s.home < s.away ? "loses" : "draws"} vs {teamB}
                </p>
              )}
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}
