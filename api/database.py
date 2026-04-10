"""Database models and session configuration for the tennis predictor API.

The project stores three kinds of data:
- players: lightweight player metadata copied from the ATP match CSVs
- elo_ratings: latest overall and surface-specific Elo snapshots per player
- matches: historical match results used for feature engineering at request time
"""

import os
from sqlalchemy import create_engine, Column, Float, Integer, String, Date
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://tennis:tennis123@localhost:5432/tennisdb")

# `use_insertmanyvalues=False` keeps inserts compatible with the local
# Postgres/SQLAlchemy setup used by this project.
engine = create_engine(DATABASE_URL, use_insertmanyvalues=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Player(Base):
    """Stored player identity and a few attributes from the source CSVs."""

    __tablename__ = "players"

    player_id   = Column(String, primary_key=True)
    name        = Column(String, nullable=False)
    hand        = Column(String)
    country     = Column(String)
    age         = Column(Float)

class EloRating(Base):
    """Latest Elo snapshot for a player across overall and surface buckets."""

    __tablename__ = "elo_ratings"

    player_id       = Column(String, primary_key=True)
    overall_elo     = Column(Float, default=1500)
    hard_elo        = Column(Float, default=1500)
    clay_elo        = Column(Float, default=1500)
    grass_elo       = Column(Float, default=1500)
    last_updated    = Column(Date)

class Match(Base):
    """Historical match result used to derive matchup features on demand."""

    __tablename__ = "matches"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    tourney_date    = Column(Date)
    tourney_name    = Column(String)
    surface         = Column(String)
    tourney_level   = Column(String)
    round           = Column(String)
    winner_id       = Column(String)
    loser_id        = Column(String)
    winner_rank     = Column(Float)
    loser_rank      = Column(Float)

def init_db():
    """Create any tables that do not already exist."""

    Base.metadata.create_all(engine)
    print("Tables created!")

if __name__ == "__main__":
    init_db()
