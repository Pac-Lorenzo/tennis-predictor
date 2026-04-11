"""Load historical ATP CSV data into Postgres and compute Elo ratings.

This script is the bridge between the raw `data/*.csv` files and the API's
database-backed runtime. Its job is to:
1. read the yearly match CSVs
2. clean and sort the rows chronologically
3. replay the full match history to build overall and surface Elo ratings
4. store players, Elo snapshots, and matches in Postgres

The script is safe to rerun for the current schema because it refreshes player
ages, updates Elo snapshots, clears the `matches` table, and then reinserts the
historical match history from scratch.
"""

import glob
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.database import SessionLocal, Player, EloRating, Match, init_db


# Standard Elo configuration used for both overall and surface-specific ratings.
K = 32
BASE_ELO = 1500
SURFACE_MAP = {"Hard": "hard", "Clay": "clay", "Grass": "grass", "Carpet": "hard"}


def expected(a, b):
    """Return the expected win probability of player A versus player B."""

    return 1 / (1 + 10 ** ((b - a) / 400))


def update_elo(winner_elo, loser_elo):
    """Apply one match result to a winner/loser Elo pair."""

    exp = expected(winner_elo, loser_elo)
    return (
        winner_elo + K * (1 - exp),
        loser_elo + K * (0 - (1 - exp)),
    )


def load_matches():
    """Read all yearly CSVs, apply cleaning rules, and sort chronologically."""

    print("Loading CSVs...")
    files = sorted(glob.glob("data/20*.csv"))
    dfs = [pd.read_csv(file_path) for file_path in files]
    df = pd.concat(dfs, ignore_index=True)
    # Drop any duplicate header rows that appear mid-file
    df = df[df["tourney_date"] != "tourney_date"]

    # Keep rows that have usable ranking information and exclude walkovers or
    # retirements so the Elo replay only contains completed matches.
    df = df.dropna(subset=["winner_rank", "loser_rank"])
    df = df[~df["score"].str.contains("W/O|RET", na=False)]
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    df = df.sort_values("tourney_date").reset_index(drop=True)

    print(f"Loaded {len(df)} matches")
    return df


def get_elo(store, player_id):
    """Return a player's current Elo from a store, defaulting to baseline."""

    return store.get(player_id, BASE_ELO)


def compute_elos(df):
    """Replay match history in chronological order to build Elo snapshots."""

    elo_overall = {}
    elo_surface = {"hard": {}, "clay": {}, "grass": {}}

    print("Computing ELO ratings...")
    for _, row in df.iterrows():
        winner_id = row["winner_id"]
        loser_id = row["loser_id"]
        surface_key = SURFACE_MAP.get(row["surface"], "hard")

        overall_winner_elo, overall_loser_elo = update_elo(
            get_elo(elo_overall, winner_id),
            get_elo(elo_overall, loser_id),
        )
        elo_overall[winner_id] = overall_winner_elo
        elo_overall[loser_id] = overall_loser_elo

        surface_winner_elo, surface_loser_elo = update_elo(
            get_elo(elo_surface[surface_key], winner_id),
            get_elo(elo_surface[surface_key], loser_id),
        )
        elo_surface[surface_key][winner_id] = surface_winner_elo
        elo_surface[surface_key][loser_id] = surface_loser_elo

    print("ELO computed!")
    return elo_overall, elo_surface


def upsert_players(db, df):
    """Insert players seen in the CSVs if they are not already present."""

    print("Writing players...")
    players_seen = {}
    for _, row in df.iterrows():
        for role, player_id in (("winner", row["winner_id"]), ("loser", row["loser_id"])):
            # Some CSV rows have a missing player_id (NaN) — skip them to avoid
            # inserting a NULL primary key.
            if pd.isna(player_id) or player_id in players_seen:
                continue

            players_seen[player_id] = True
            existing = db.query(Player).filter_by(player_id=player_id).first()
            if existing:
                continue

            db.add(
                Player(
                    player_id=player_id,
                    name=row[f"{role}_name"],
                    hand=row.get(f"{role}_hand"),
                    country=row.get(f"{role}_ioc"),
                    age=row.get(f"{role}_age"),
                )
            )

    db.commit()
    print(f"Wrote {len(players_seen)} players")


def refresh_player_ages(db, df):
    """Update each player's stored age from the most recent rows in the CSVs."""

    print("Updating player ages...")
    age_map = {}
    for _, row in df.iterrows():
        for role, pid in (("winner", row["winner_id"]), ("loser", row["loser_id"])):
            age = row.get(f"{role}_age")
            if pid and age:
                age_map[pid] = age
    for player_id, age in age_map.items():
        if pd.isna(player_id):
            continue
        db.query(Player).filter_by(player_id=player_id).update({"age": age})
    db.commit()
    print(f"Updated ages for {len(age_map)} players")

def upsert_elo_ratings(db, elo_overall, elo_surface):
    """Store the latest Elo snapshot for every player seen in the match data."""

    print("Writing ELO ratings...")
    today = datetime.today().date()
    all_players = set(elo_overall.keys())

    # Filter out NaN player IDs that can slip in from malformed CSV rows.
    # `p == p` is False for NaN floats — cheaper than importing math.isnan.
    all_players = {p for p in all_players if isinstance(p, str) and p == p}
    for player_id in all_players:
        existing = db.query(EloRating).filter_by(player_id=player_id).first()
        if existing:
            existing.overall_elo = elo_overall.get(player_id, BASE_ELO)
            existing.hard_elo = elo_surface["hard"].get(player_id, BASE_ELO)
            existing.clay_elo = elo_surface["clay"].get(player_id, BASE_ELO)
            existing.grass_elo = elo_surface["grass"].get(player_id, BASE_ELO)
            existing.last_updated = today
            continue

        db.add(
            EloRating(
                player_id=player_id,
                overall_elo=elo_overall.get(player_id, BASE_ELO),
                hard_elo=elo_surface["hard"].get(player_id, BASE_ELO),
                clay_elo=elo_surface["clay"].get(player_id, BASE_ELO),
                grass_elo=elo_surface["grass"].get(player_id, BASE_ELO),
                last_updated=today,
            )
        )

    db.commit()
    print(f"Wrote ELO for {len(all_players)} players")


def insert_matches(db, df):
    """Insert raw historical matches used later for feature engineering."""

    print("Writing matches...")
    db.query(Match).delete()
    db.commit()

    records = [
        Match(
            tourney_date=row["tourney_date"].date(),
            tourney_name=row["tourney_name"],
            surface=row["surface"],
            tourney_level=str(row["tourney_level"]),
            round=row["round"],
            winner_id=row["winner_id"],
            loser_id=row["loser_id"],
            winner_rank=row["winner_rank"],
            loser_rank=row["loser_rank"],
        )
        for _, row in df.iterrows()
    ]

    CHUNK = 500
    for i in range(0, len(records), CHUNK):
        db.bulk_save_objects(records[i:i+CHUNK])
        db.commit()
        print(f"  {min(i+CHUNK, len(records))}/{len(records)} matches written...")

    print(f"Wrote {len(df)} matches")

def main():
    """Run the full import pipeline from CSVs to database tables."""

    df = load_matches()
    elo_overall, elo_surface = compute_elos(df)

    init_db()
    db = SessionLocal()
    try:
        upsert_players(db, df)
        refresh_player_ages(db, df)
        upsert_elo_ratings(db, elo_overall, elo_surface)
        insert_matches(db, df)
    finally:
        db.close()

    print("\nDone! Database is ready.")


if __name__ == "__main__":
    main()
