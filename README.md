# Tennis Predictor

A FastAPI service that predicts ATP match outcomes from historical match data, Elo ratings, and a trained scikit-learn model.

## What The Project Does

The repository combines three layers:

1. Historical ATP match CSVs in `data/`
2. A database-backed feature layer built from those CSVs
3. A FastAPI app that assembles matchup features and returns win probabilities

At a high level, the workflow is:

1. `scripts/load_data.py` reads the CSV files, cleans the raw rows, and replays match history chronologically.
2. During that replay, the script computes both overall Elo and surface-specific Elo for each player.
3. The script stores player metadata, refreshes player ages, updates Elo snapshots, and reloads the historical matches in Postgres.
4. `api/main.py` looks up two requested players, derives model features from the stored data, and sends those features to the trained model in `api/model.py`.
5. The API returns player win probabilities, the predicted winner, a simple confidence label, and each player's Elo.

## Project Structure

- `api/main.py`: FastAPI app, request handling, feature assembly, and prediction endpoints
- `api/model.py`: loads the trained model and feature column order, then runs `predict_proba`
- `api/database.py`: SQLAlchemy engine, session factory, and table models
- `scripts/load_data.py`: imports CSV data into Postgres and computes Elo ratings
- `model/`: trained model artifacts and saved feature order
- `data/`: raw ATP match CSV files used to populate the database
- `notebooks/`: exploratory work and model experimentation

## API Flow

The `/predict` endpoint works like this:

1. Resolve `player1_name` and `player2_name` to internal `player_id` values.
2. Fetch each player's overall Elo and requested-surface Elo from `elo_ratings`.
3. Compute recent win rate, surface-specific recent win rate, head-to-head history, and days since last match from the `matches` table.
4. Pull the latest available ranking value for each player from their most recent stored match.
5. Pull stored player ages from the `players` table, using a neutral fallback only when age is missing.
6. Encode match context such as surface, tournament level, round, indoor flag, and best-of format.
7. Build the full feature vector in the exact order saved in `model/feature_cols.json`.
8. Run the scikit-learn model and return probabilities plus a confidence label derived from the probability gap.

## Data Loading Flow

`scripts/load_data.py` currently performs a full refresh of the database-backed historical layer:

1. Load all yearly CSV files matching `data/20*.csv`
2. Drop rows missing winner or loser rank
3. Exclude walkovers and retirements
4. Sort matches by tournament date
5. Recompute Elo ratings from oldest match to newest match
6. Insert missing players
7. Refresh each player's age using the latest rows seen in the source data
8. Upsert each player's latest Elo snapshot
9. Clear the `matches` table
10. Reinsert all historical matches

## Running Locally

The app expects a Postgres database at:

```text
postgresql://tennis:tennis123@localhost:5432/tennisdb
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn api.main:app --reload
```

Load the database from the CSV files:

```bash
python scripts/load_data.py
```

## Docker

Build:

```bash
docker build -t tennis-predictor .
```

Run:

```bash
docker run -p 8000:8000 tennis-predictor
```

The Docker image installs the dependencies from `requirements.txt`, copies the API code and model artifacts, and starts Uvicorn on port `8000`.

## Current Assumptions And Limitations

- The API uses Elo values as the `rank_points` model features because that is how this version of the feature set is wired.
- `age_diff` is still hardcoded to a neutral value in the API.
- Rolling 7-day and 30-day match-count features are currently hardcoded to zero in the API.
- Player matching is based on a case-insensitive partial-name search and returns the first match found.
