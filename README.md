cat > README.md << 'EOF'
# Tennis Match Predictor API

A production-ready REST API that predicts ATP tennis match outcomes using machine learning.

## Model Performance
- **Accuracy:** 80.8% on held-out test set
- **AUC:** 0.91
- **Algorithm:** HistGradientBoostingClassifier
- **Training data:** 11 years of ATP matches (2015-2025), 29,000+ matches

## Features (47 total)
- ELO ratings (overall + surface-specific)
- Head-to-head win rates (overall + per surface)
- Recent form (last 10 matches)
- Surface-specific win rates
- Days rest / match load (fatigue)
- ATP ranking + ranking points
- Tournament level + round

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict match outcome |
| GET | `/players?search=` | Search players |
| GET | `/players/{name}/elo` | Get player ELO ratings |
| GET | `/h2h?player1=&player2=` | Head-to-head stats |
| GET | `/health` | Health check |

## Example Request

```bash
curl -X POST https://your-api-url/predict \
  -H "Content-Type: application/json" \
  -d '{
    "player1_name": "Djokovic",
    "player2_name": "Alcaraz",
    "surface": "Clay",
    "tournament_level": "G",
    "round": "F",
    "best_of": 5,
    "indoor": false
  }'
```

## Example Response
```json
{
  "player1_win_probability": 0.972,
  "player2_win_probability": 0.028,
  "predicted_winner": "Novak Djokovic",
  "confidence": "high",
  "player1_elo": 2022.5,
  "player2_elo": 2139.3
}
```

## Tech Stack
- **ML:** scikit-learn, pandas, numpy
- **API:** FastAPI, Pydantic
- **Database:** PostgreSQL (SQLAlchemy ORM)
- **Container:** Docker
- **Deployment:** AWS ECS Fargate + RDS

## Local Setup

```bash
# Start PostgreSQL
docker run --name tennis-db \
  -e POSTGRES_USER=tennis \
  -e POSTGRES_PASSWORD=tennis123 \
  -e POSTGRES_DB=tennisdb \
  -p 5432:5432 -d postgres:15

# Install dependencies
pip install -r requirements.txt

# Load data into DB
python scripts/load_data.py

# Run API
uvicorn api.main:app --reload --port 8001
```

## Project Structure
tennis-predictor/
├── api/
│   ├── main.py        # FastAPI app + endpoints
│   ├── model.py       # Model loading + inference
│   └── database.py    # SQLAlchemy models + DB connection
├── model/
│   ├── tennis_model_v2.pkl
│   └── feature_cols.json
├── scripts/
│   └── load_data.py   # DB bootstrap script
├── notebooks/         # EDA + model training
├── Dockerfile
└── requirements.txt
