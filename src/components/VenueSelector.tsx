"use client";

import { motion } from "framer-motion";
import type { VenueType } from "@/lib/types";

interface Props {
  venue: VenueType;
  onChange: (venue: VenueType) => void;
  teamA: string;
  teamB: string;
}

const options: { id: VenueType; label: (a: string, b: string) => string; icon: string }[] = [
  { id: "teamA_home", label: (a) => `${a || "Team A"} Home`, icon: "🏟" },
  { id: "neutral", label: () => "Neutral Venue", icon: "🌍" },
  { id: "teamB_home", label: (_, b) => `${b || "Team B"} Home`, icon: "🏟" },
];

export default function VenueSelector({ venue, onChange, teamA, teamB }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
      className="space-y-2"
    >
      <label className="block text-xs font-medium uppercase tracking-widest text-muted-light">
        Match Venue
      </label>
      <div className="flex flex-col gap-2 sm:flex-row">
        {options.map((opt) => {
          const active = venue === opt.id;
          return (
            <button
              key={opt.id}
              type="button"
              onClick={() => onChange(opt.id)}
              className={`group relative flex-1 rounded-xl px-4 py-3 text-left text-sm transition-all duration-300 ${
                active
                  ? "glass-panel glow-emerald text-white"
                  : "border border-white/8 bg-surface-raised/40 text-muted-light hover:border-white/15 hover:text-white"
              }`}
            >
              <span className="mr-2">{opt.icon}</span>
              {opt.label(teamA, teamB)}
              {active && (
                <motion.div
                  layoutId="venue-indicator"
                  className="absolute inset-0 rounded-xl border border-accent-emerald/30"
                  transition={{ type: "spring", stiffness: 400, damping: 30 }}
                />
              )}
            </button>
          );
        })}
      </div>
    </motion.div>
  );
}
