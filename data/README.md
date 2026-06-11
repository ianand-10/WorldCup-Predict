# Data Files

Place your international football CSV files here.

## matches.csv

Match results from 2021 to present.

| Column | Required | Example |
|--------|----------|---------|
| `date` | Yes | `2024-06-15` |
| `home_team` | Yes | `Brazil` |
| `away_team` | Yes | `Argentina` |
| `home_score` | Yes | `2` |
| `away_score` | Yes | `1` |
| `tournament` | Yes | `Friendly` |
| `city` | Yes | `Maracana` |
| `country` | Yes | `Brazil` |
| `neutral` | Yes | `FALSE` or `TRUE` |

## goalscorers.csv

Per-goal scorer records (one row per goal).

| Column | Required | Example |
|--------|----------|---------|
| `date` | Yes | `2024-06-15` |
| `home_team` | Yes | `Brazil` |
| `away_team` | Yes | `Argentina` |
| `team` | Yes | `Brazil` |
| `scorer` | Yes | `Neymar` |
| `minute` | Yes | `34` |
| `own_goal` | Yes | `FALSE` |
| `penalty` | Yes | `FALSE` |

## fifa_rankings.csv (optional but recommended)

Current FIFA rankings used as an ML feature. Update this file whenever rankings change.

| Column | Required | Example |
|--------|----------|---------|
| `team` | Yes | `Brazil` |
| `rank` | Yes* | `5` |
| `points` | Yes* | `1790 |

\* Provide `points` if you have them (best). If only `rank` is available, the model converts rank to an approximate points value.

Team names must match `matches.csv` exactly. Teams not listed fall back to a default rating of 1500 points.

## Rebuilding the model

After updating any CSV, regenerate the prediction data:

```bash
pip install -r scripts/requirements.txt
npm run build:data
```

This writes:

- `public/data/teams.json` — ELO ratings
- `public/data/scorers.json` — goalscorer shares
- `public/data/liveState.json` — form, rest days, head-to-head, FIFA points
- `public/data/model.json` — blend weights, Dixon-Coles rho, ML metrics

Commit all of the above and push to redeploy GitHub Pages.

## Team name consistency

Team names must match exactly across all files (e.g. always `United States`, never `USA` in one file and `United States` in another).
