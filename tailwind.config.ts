import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: "#0A0A0F",
          raised: "#12121A",
          glass: "rgba(18, 18, 26, 0.72)",
        },
        accent: {
          emerald: "#10B981",
          gold: "#F5C542",
          glow: "rgba(16, 185, 129, 0.35)",
        },
        muted: {
          DEFAULT: "#71717A",
          light: "#A1A1AA",
        },
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "monospace"],
      },
      animation: {
        "mesh-shift": "meshShift 18s ease-in-out infinite alternate",
        shimmer: "shimmer 2.5s ease-in-out infinite",
        "orbit-sweep": "orbitSweep 8s linear infinite",
        "pulse-glow": "pulseGlow 3s ease-in-out infinite",
      },
      keyframes: {
        meshShift: {
          "0%": { transform: "translate(0%, 0%) scale(1)" },
          "50%": { transform: "translate(3%, -2%) scale(1.05)" },
          "100%": { transform: "translate(-2%, 3%) scale(1.02)" },
        },
        shimmer: {
          "0%, 100%": { opacity: "0.6" },
          "50%": { opacity: "1" },
        },
        orbitSweep: {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" },
        },
        pulseGlow: {
          "0%, 100%": { boxShadow: "0 0 20px rgba(16, 185, 129, 0.2)" },
          "50%": { boxShadow: "0 0 40px rgba(245, 197, 66, 0.35)" },
        },
      },
      backdropBlur: {
        glass: "16px",
      },
    },
  },
  plugins: [],
};

export default config;
