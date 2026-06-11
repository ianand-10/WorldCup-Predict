"use client";

import { motion } from "framer-motion";
import type { GoalscorerPrediction } from "@/lib/types";

interface Props {
  team: string;
  scorers: GoalscorerPrediction[];
  accent: "emerald" | "gold";
}

export default function GoalscorerPills({ team, scorers, accent }: Props) {
  const barColor = accent === "emerald" ? "bg-accent-emerald" : "bg-accent-gold";
  const textColor = accent === "emerald" ? "text-accent-emerald" : "text-accent-gold";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.25 }}
      className="space-y-3"
    >
      <h3 className="text-xs font-medium uppercase tracking-widest text-muted-light">
        Likely Scorers — {team}
      </h3>

      {scorers.length === 0 ? (
        <p className="rounded-xl border border-white/8 bg-surface-raised/40 px-4 py-3 text-sm text-muted">
          Insufficient scorer data for this team.
        </p>
      ) : (
        <div className="space-y-2">
          {scorers.map((s, i) => {
            const maxProb = scorers[0].probability;
            const widthPct = (s.probability / maxProb) * 100;

            return (
              <motion.div
                key={s.name}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                className="group glass-panel rounded-xl px-4 py-3 transition-all duration-300 hover:border-white/15"
              >
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-sm font-medium text-white group-hover:text-white">
                    {s.name}
                  </span>
                  <span className={`font-mono text-xs ${textColor}`}>
                    {(s.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="relative h-1.5 overflow-hidden rounded-full bg-white/5">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${widthPct}%` }}
                    transition={{ duration: 0.5, delay: i * 0.05 }}
                    className={`absolute inset-y-0 left-0 rounded-full ${barColor} opacity-80`}
                  />
                </div>
                <p className="mt-1 font-mono text-[10px] text-muted">
                  {s.goals} goals since 2021
                </p>
              </motion.div>
            );
          })}
        </div>
      )}
    </motion.div>
  );
}
