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

## Rebuilding the model

After updating either CSV, regenerate the prediction data:

```bash
pip install -r scripts/requirements.txt
npm run build:data
```

This writes updated ratings to `public/data/`.

## Team name consistency

Team names must match exactly between both files (e.g. always `United States`, never `USA` in one file and `United States` in another).
