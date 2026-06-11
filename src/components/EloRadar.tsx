"use client";

import { motion } from "framer-motion";
import type { TeamRatings } from "@/lib/types";

interface Props {
  teamA: string;
  teamB: string;
  ratingsA: TeamRatings;
  ratingsB: TeamRatings;
}

function normalize(val: number, min: number, max: number): number {
  return Math.max(0.15, Math.min(1, (val - min) / (max - min)));
}

function RadarChart({
  ratings,
  color,
  label,
}: {
  ratings: TeamRatings;
  color: string;
  label: string;
}) {
  const min = 1300;
  const max = 1700;
  const axes = [
    { key: "overall", label: "OVR" },
    { key: "offense", label: "ATK" },
    { key: "defense", label: "DEF" },
  ] as const;

  const cx = 60;
  const cy = 60;
  const maxR = 45;

  const values = axes.map((a) => normalize(ratings[a.key], min, max));

  const points = values
    .map((v, i) => {
      const angle = (i * 2 * Math.PI) / axes.length - Math.PI / 2;
      const r = v * maxR;
      return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
    })
    .join(" ");

  const gridLevels = [0.33, 0.66, 1];

  return (
    <div className="flex flex-col items-center">
      <svg width="120" height="120" viewBox="0 0 120 120">
        {gridLevels.map((level) => (
          <polygon
            key={level}
            points={axes
              .map((_, i) => {
                const angle = (i * 2 * Math.PI) / axes.length - Math.PI / 2;
                const r = level * maxR;
                return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
              })
              .join(" ")}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="1"
          />
        ))}

        {axes.map((a, i) => {
          const angle = (i * 2 * Math.PI) / axes.length - Math.PI / 2;
          const x = cx + maxR * Math.cos(angle);
          const y = cy + maxR * Math.sin(angle);
          return (
            <g key={a.key}>
              <line x1={cx} y1={cy} x2={x} y2={y} stroke="rgba(255,255,255,0.06)" />
              <text
                x={cx + (maxR + 12) * Math.cos(angle)}
                y={cy + (maxR + 12) * Math.sin(angle)}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="rgba(255,255,255,0.4)"
                fontSize="8"
              >
                {a.label}
              </text>
            </g>
          );
        })}

        <motion.polygon
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 0.7, scale: 1 }}
          transition={{ duration: 0.5 }}
          points={points}
          fill={color}
          fillOpacity={0.25}
          stroke={color}
          strokeWidth="1.5"
          style={{ transformOrigin: `${cx}px ${cy}px` }}
        />
      </svg>
      <p className="mt-1 max-w-[100px] truncate text-center text-[10px] text-muted-light">
        {label}
      </p>
      <p className="font-mono text-xs text-white">{Math.round(ratings.overall)}</p>
    </div>
  );
}

export default function EloRadar({ teamA, teamB, ratingsA, ratingsB }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.15 }}
      className="glass-panel rounded-2xl p-5"
    >
      <h3 className="mb-4 text-xs font-medium uppercase tracking-widest text-muted-light">
        ELO Radar Duel
      </h3>
      <div className="flex items-center justify-around">
        <RadarChart ratings={ratingsA} color="#10B981" label={teamA} />
        <div className="text-center text-muted text-xs font-mono">VS</div>
        <RadarChart ratings={ratingsB} color="#F5C542" label={teamB} />
      </div>
    </motion.div>
  );
}
