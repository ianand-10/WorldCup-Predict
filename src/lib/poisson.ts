export function factorial(n: number): number {
  if (n <= 1) return 1;
  let result = 1;
  for (let i = 2; i <= n; i++) result *= i;
  return result;
}

export function poissonPMF(k: number, lambda: number): number {
  if (lambda <= 0) return k === 0 ? 1 : 0;
  return (Math.exp(-lambda) * Math.pow(lambda, k)) / factorial(k);
}

/** Dixon-Coles adjustment for low-scoring correlation */
export function dixonColesTau(
  homeGoals: number,
  awayGoals: number,
  lambdaHome: number,
  lambdaAway: number,
  rho: number
): number {
  if (homeGoals === 0 && awayGoals === 0) {
    return 1 - lambdaHome * lambdaAway * rho;
  }
  if (homeGoals === 0 && awayGoals === 1) {
    return 1 + lambdaAway * rho;
  }
  if (homeGoals === 1 && awayGoals === 0) {
    return 1 + lambdaHome * rho;
  }
  if (homeGoals === 1 && awayGoals === 1) {
    return 1 - rho;
  }
  return 1;
}

export function scorelineMatrix(
  lambdaHome: number,
  lambdaAway: number,
  maxGoals: number = 8,
  rho: number = 0
): { home: number; away: number; probability: number }[] {
  const scorelines: { home: number; away: number; probability: number }[] = [];

  for (let h = 0; h <= maxGoals; h++) {
    for (let a = 0; a <= maxGoals; a++) {
      const tau = dixonColesTau(h, a, lambdaHome, lambdaAway, rho);
      const prob =
        Math.max(0, tau) * poissonPMF(h, lambdaHome) * poissonPMF(a, lambdaAway);
      scorelines.push({ home: h, away: a, probability: prob });
    }
  }

  const total = scorelines.reduce((s, x) => s + x.probability, 0);
  return scorelines
    .map((s) => ({ ...s, probability: s.probability / total }))
    .sort((a, b) => b.probability - a.probability);
}

export function outcomeProbabilities(
  scorelines: { home: number; away: number; probability: number }[],
  perspectiveHome: boolean
): { winA: number; draw: number; winB: number } {
  let winHome = 0;
  let draw = 0;
  let winAway = 0;

  for (const s of scorelines) {
    if (s.home > s.away) winHome += s.probability;
    else if (s.home < s.away) winAway += s.probability;
    else draw += s.probability;
  }

  if (perspectiveHome) {
    return { winA: winHome, draw, winB: winAway };
  }
  return { winA: winAway, draw, winB: winHome };
}
