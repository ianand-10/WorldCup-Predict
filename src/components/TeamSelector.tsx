"use client";

import { motion } from "framer-motion";

interface Props {
  label: string;
  teams: string[];
  value: string;
  onChange: (team: string) => void;
  accent?: "emerald" | "gold";
  exclude?: string;
}

export default function TeamSelector({
  label,
  teams,
  value,
  onChange,
  accent = "emerald",
  exclude,
}: Props) {
  const filtered = exclude
    ? teams.filter((t) => t !== exclude)
    : teams;

  const ringColor =
    accent === "emerald"
      ? "focus:ring-accent-emerald/50"
      : "focus:ring-accent-gold/50";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-2"
    >
      <label className="block text-xs font-medium uppercase tracking-widest text-muted-light">
        {label}
      </label>
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className={`w-full appearance-none rounded-xl border border-white/10 bg-surface-raised/80 px-4 py-3.5 pr-10 text-sm text-white outline-none transition-all duration-300 hover:border-white/20 focus:border-accent-emerald/40 focus:ring-2 ${ringColor}`}
        >
          <option value="" disabled>
            Select team...
          </option>
          {filtered.map((team) => (
            <option key={team} value={team} className="bg-surface-raised">
              {team}
            </option>
          ))}
        </select>
        <div className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-muted">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" fill="none" />
          </svg>
        </div>
      </div>
    </motion.div>
  );
}
