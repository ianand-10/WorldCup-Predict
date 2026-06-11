# World Cup Predictor

Premium international football match predictor powered by multi-dimensional ELO ratings, Poisson scoreline modeling, and ML-calibrated outcomes.

![Stack](https://img.shields.io/badge/Next.js-15-black) ![Stack](https://img.shields.io/badge/GitHub_Pages-ready-green)

## Features

- **Multi-dimensional ELO** — overall, offensive, defensive, and venue-specific ratings
- **Poisson scorelines** — top 15 most likely scorelines with probabilities
- **ML calibration** — logistic regression blended with ELO predictions
- **Goalscorer predictions** — likely scorers based on historical goal share since 2021
- **Venue-aware** — home, away, or neutral venue selection
- **Premium UI** — dark glassmorphism design with animated backgrounds and micro-interactions

## Deploy to GitHub Pages (one-time setup)

### 1. Push this repo to GitHub

```bash
git init
git add .
git commit -m "Initial World Cup Predictor"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

### 2. Enable GitHub Pages

1. Open your repo on GitHub → **Settings** → **Pages**
2. Under **Build and deployment**, set **Source** to **GitHub Actions**
3. Push to `main` (or run the **Deploy to GitHub Pages** workflow manually)

The workflow in `.github/workflows/deploy-pages.yml` builds the static site and publishes it automatically.

Your site will be live at:

```
https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/
```

> **Note:** If your repo is named `YOUR_USERNAME.github.io` (a user/organization site), set `GITHUB_PAGES=false` in the deploy workflow — the site is served from the root URL with no subpath.

## Local Development

Requires [Node.js 18+](https://nodejs.org/) and Python 3.10+ (only for rebuilding model data).

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Your CSV Files

Your data is already in place:

| File | Location |
|------|----------|
| Match results | `data/matches.csv` |
| Goalscorers | `data/goalscorers.csv` |

See [`data/README.md`](data/README.md) for the full column schema.

### Updating data

1. Replace `data/matches.csv` and/or `data/goalscorers.csv`
2. Rebuild the model:

```bash
pip install -r scripts/requirements.txt
npm run build:data
```

3. Commit the updated `public/data/` files and push to GitHub — Pages redeploys automatically.

You can also trigger **Rebuild Model Data** from the Actions tab (runs when `data/` or `scripts/` change).

## How the Model Works

1. **ELO ratings** are computed chronologically from 2021 match results, with separate offensive/defensive/venue dimensions
2. **Expected goals (xG)** are derived from offensive vs defensive ELO differentials plus home advantage
3. **Poisson distribution** generates scoreline probabilities from xG values
4. **Logistic regression ML** (trained on point-in-time features) calibrates win/draw/loss probabilities
5. **Final output** = 65% ELO/Poisson + 35% ML blend
6. **Goalscorers** = team xG × player's historical goal share

## Project Structure

```
├── data/                  # Your CSV files
├── public/data/           # Generated ratings (committed for GitHub Pages)
├── scripts/build_model.py # ELO + ML training pipeline
├── src/
│   ├── app/               # Next.js pages
│   ├── components/        # UI components
│   └── lib/               # Prediction engine (runs in the browser)
└── .github/workflows/
    ├── deploy-pages.yml   # Builds & publishes to GitHub Pages
    └── build-data.yml     # Rebuilds model JSON from CSVs
```

## Tech Stack

- **Frontend:** Next.js 15 (static export), React 19, Tailwind CSS, Framer Motion
- **Model:** Python (pandas, scikit-learn) for offline training; TypeScript runtime in the browser
- **Deploy:** GitHub Pages via GitHub Actions
