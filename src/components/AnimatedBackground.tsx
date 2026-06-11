"use client";

import { motion } from "framer-motion";
import type { VenueType } from "@/lib/types";

interface Props {
  venue: VenueType;
}

const venueColors: Record<VenueType, { a: string; b: string; c: string }> = {
  teamA_home: { a: "#1a1408", b: "#0f1a14", c: "#2a1f0a" },
  neutral: { a: "#0a1218", b: "#0a0f1a", c: "#0f1a18" },
  teamB_home: { a: "#0a0f1a", b: "#0f1218", c: "#0a1418" },
};

export default function AnimatedBackground({ venue }: Props) {
  const colors = venueColors[venue];

  return (
    <div className="fixed inset-0 -z-10 overflow-hidden">
      <motion.div
        key={venue}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.2 }}
        className="absolute inset-0"
        style={{
          background: `radial-gradient(ellipse 80% 60% at 20% 30%, ${colors.c}88 0%, transparent 60%),
                       radial-gradient(ellipse 70% 50% at 80% 70%, ${colors.b}66 0%, transparent 55%),
                       radial-gradient(ellipse 60% 40% at 50% 50%, ${colors.a} 0%, #0A0A0F 70%)`,
        }}
      />

      <div
        className="absolute inset-0 opacity-[0.04]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='1'%3E%3Cpath d='M30 0l4 8h8l-6 6 2 8-8-5-8 5 2-8-6-6h8z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          backgroundSize: "60px 60px",
        }}
      />

      <motion.div
        animate={{ x: [0, 30, -20, 0], y: [0, -20, 15, 0] }}
        transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
        className="absolute -top-1/4 -left-1/4 h-[60vh] w-[60vh] rounded-full opacity-20 blur-[100px]"
        style={{ background: "radial-gradient(circle, #10B981 0%, transparent 70%)" }}
      />
      <motion.div
        animate={{ x: [0, -25, 20, 0], y: [0, 25, -15, 0] }}
        transition={{ duration: 22, repeat: Infinity, ease: "easeInOut" }}
        className="absolute -bottom-1/4 -right-1/4 h-[50vh] w-[50vh] rounded-full opacity-15 blur-[90px]"
        style={{ background: "radial-gradient(circle, #F5C542 0%, transparent 70%)" }}
      />
    </div>
  );
}
