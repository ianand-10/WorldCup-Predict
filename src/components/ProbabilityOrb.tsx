"use client";

import { motion } from "framer-motion";

interface Props {
  teamA: string;
  teamB: string;
  winA: number;
  draw: number;
  winB: number;
  expectedA: number;
  expectedB: number;
}

function Arc({
  startAngle,
  sweep,
  color,
  radius,
}: {
  startAngle: number;
  sweep: number;
  color: string;
  radius: number;
}) {
  const cx = 120;
  const cy = 120;
  const start = polarToCartesian(cx, cy, radius, startAngle);
  const end = polarToCartesian(cx, cy, radius, startAngle + sweep);
  const largeArc = sweep > 180 ? 1 : 0;

  const d = [
    "M", start.x, start.y,
    "A", radius, radius, 0, largeArc, 1, end.x, end.y,
  ].join(" ");

  return (
    <path
      d={d}
      fill="none"
      stroke={color}
      strokeWidth="10"
      strokeLinecap="round"
      opacity={0.9}
    />
  );
}

function polarToCartesian(cx: number, cy: number, r: number, angle: number) {
  const rad = ((angle - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

export default function ProbabilityOrb({
  teamA,
  teamB,
  winA,
  draw,
  winB,
  expectedA,
  expectedB,
}: Props) {
  const total = winA + draw + winB;
  const wA = (winA / total) * 360;
  const wD = (draw / total) * 360;
  const wB = (winB / total) * 360;

  const leader =
    winA >= winB ? { name: teamA, prob: winA } : { name: teamB, prob: winB };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className="relative flex flex-col items-center"
    >
      <div className="relative">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
          className="absolute inset-0 rounded-full opacity-30"
          style={{
            background:
              "conic-gradient(from 0deg, transparent, #10B98144, transparent, #F5C54244, transparent)",
          }}
        />

        <svg width="240" height="240" viewBox="0 0 240 240" className="relative z-10">
          <circle cx="120" cy="120" r="90" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="10" />
          <Arc startAngle={0} sweep={wA} color="#10B981" radius={90} />
          <Arc startAngle={wA} sweep={wD} color="#71717A" radius={90} />
          <Arc startAngle={wA + wD} sweep={wB} color="#F5C542" radius={90} />
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          <p className="text-[10px] uppercase tracking-widest text-muted">Favorite</p>
          <p className="mt-1 max-w-[120px] truncate text-sm font-semibold text-white">
            {leader.name}
          </p>
          <p className="font-mono text-2xl font-bold text-gradient-gold">
            {(leader.prob * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="mt-6 grid w-full grid-cols-3 gap-3 text-center">
        <div className="glass-panel rounded-xl px-3 py-3">
          <p className="truncate text-[10px] uppercase tracking-wider text-muted">{teamA}</p>
          <p className="font-mono text-lg font-semibold text-accent-emerald">
            {(winA * 100).toFixed(1)}%
          </p>
          <p className="font-mono text-xs text-muted-light">xG {expectedA}</p>
        </div>
        <div className="glass-panel rounded-xl px-3 py-3">
          <p className="text-[10px] uppercase tracking-wider text-muted">Draw</p>
          <p className="font-mono text-lg font-semibold text-muted-light">
            {(draw * 100).toFixed(1)}%
          </p>
        </div>
        <div className="glass-panel rounded-xl px-3 py-3">
          <p className="truncate text-[10px] uppercase tracking-wider text-muted">{teamB}</p>
          <p className="font-mono text-lg font-semibold text-accent-gold">
            {(winB * 100).toFixed(1)}%
          </p>
          <p className="font-mono text-xs text-muted-light">xG {expectedB}</p>
        </div>
      </div>
    </motion.div>
  );
}
