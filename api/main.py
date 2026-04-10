"""FastAPI application for tennis matchup lookups and match prediction.

Request-time flow:
1. Resolve the two requested players from the local database.
2. Pull their latest Elo values plus historical match information.
3. Convert that history into the exact feature set expected by the trained model.
4. Run the model and return probabilities with a lightweight confidence label.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.model import predict
from api.database import SessionLocal, Player, EloRating, Match
from sqlalchemy import func
from datetime import date

app = FastAPI(
    title="Tennis Match Predictor",
    description="Predicts ATP match outcomes using ML",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response schemas ──────────────────────────────────
class MatchRequest(BaseModel):
    """User-supplied description of an upcoming match to score."""

    player1_name: str
    player2_name: str
    surface: str        # "Hard", "Clay", "Grass"
    tournament_level: str  # "250", "500", "M", "G"
    round: str          # "R32", "QF", "SF", "F"
    best_of: int        # 3 or 5
    indoor: bool = False

class PlayerInfo(BaseModel):
    """Basic player view for future API responses."""

    player_id: str
    name: str
    overall_elo: float
    surface_elo: float

class MatchResponse(BaseModel):
    """Model output returned by the `/predict` endpoint."""

    player1_win_probability: float
    player2_win_probability: float
    predicted_winner: str
    confidence: str
    player1_elo: float
    player2_elo: float

# ── Encoding maps ─────────────────────────────────────────────
SURFACE_MAP = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 0}
LEVEL_MAP = {'250': 0, '500': 1, 'A': 2, 'M': 3, 'G': 4, 'D': 5, 'O': 5, 'F': 6}
ROUND_MAP = {
    'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
    'QF': 5, 'SF': 6, 'F': 7, 'RR': 3, 'BR': 6
}

# ── Helper functions ──────────────────────────────────────────
def find_player(db, name: str):
    """Find the first player whose name contains the provided search string."""

    player = db.query(Player).filter(
        func.lower(Player.name).contains(name.lower())
    ).first()
    if not player:
        raise HTTPException(
            status_code=404,
            detail=f"Player '{name}' not found. Try a partial name like 'Djokovic'."
        )
    return player

def get_elo(db, player_id: str, surface: str):
    """Return overall Elo plus the Elo for the requested surface."""

    elo = db.query(EloRating).filter_by(player_id=player_id).first()
    if not elo:
        return 1500.0, 1500.0
    surface_elo_map = {
        'Hard': elo.hard_elo,
        'Clay': elo.clay_elo,
        'Grass': elo.grass_elo,
        'Carpet': elo.hard_elo
    }
    return elo.overall_elo, surface_elo_map.get(surface, elo.overall_elo)

def get_recent_wr(db, player_id: str, surface: str = None, n: int = 10):
    """Compute a player's recent win rate over the latest `n` matches."""

    matches = db.query(Match).filter(
        (Match.winner_id == player_id) | (Match.loser_id == player_id)
    )
    if surface:
        matches = matches.filter(Match.surface == surface)
    matches = matches.order_by(Match.tourney_date.desc()).limit(n).all()

    if not matches:
        return 0.5, 0

    wins = sum(1 for m in matches if m.winner_id == player_id)
    return wins / len(matches), len(matches)

def get_h2h(db, p1_id: str, p2_id: str, surface: str = None):
    """Return player1's head-to-head split against player2."""

    matches = db.query(Match).filter(
        ((Match.winner_id == p1_id) & (Match.loser_id == p2_id)) |
        ((Match.winner_id == p2_id) & (Match.loser_id == p1_id))
    )
    if surface:
        matches = matches.filter(Match.surface == surface)
    matches = matches.all()

    if not matches:
        return 0.5, 0.5, 0

    p1_wins = sum(1 for m in matches if m.winner_id == p1_id)
    total = len(matches)
    return p1_wins / total, 1 - (p1_wins / total), total

def get_days_rest(db, player_id: str):
    """Estimate days since the player's last recorded match.

    The value is capped to avoid a very large layoff dominating the feature
    space for players who have been inactive for a long time.
    """

    last = db.query(Match).filter(
        (Match.winner_id == player_id) | (Match.loser_id == player_id)
    ).order_by(Match.tourney_date.desc()).first()

    if not last:
        return 7
    delta = (date.today() - last.tourney_date).days
    return min(delta, 365)

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    """Simple status endpoint for a quick smoke test."""

    return {"status": "ok", "version": "2.0.0"}

@app.get("/health")
def health():
    """Health endpoint used by local checks or container orchestration."""

    return {"status": "healthy"}

@app.get("/debug/{player_name}")
def debug_player(player_name: str, surface: str = "Clay"):
    """Return raw feature values for a single player — useful for sanity-checking the database."""

    db = SessionLocal()
    try:
        player = find_player(db, player_name)
        wr, n = get_recent_wr(db, player.player_id)
        surf_wr, surf_n = get_recent_wr(db, player.player_id, surface)
        rest = get_days_rest(db, player.player_id)
        elo_overall, elo_surf = get_elo(db, player.player_id, surface)

        return {
            "name": player.name,
            "recent_wr": wr,
            "recent_n": n,
            f"{surface}_wr": surf_wr,
            f"{surface}_n": surf_n,
            "days_rest": rest,
            "overall_elo": round(elo_overall, 1),
            f"{surface}_elo": round(elo_surf, 1)
        }
    finally:
        db.close()

        
@app.get("/players")
def list_players(search: str = ""):
    """List players, optionally filtered by a case-insensitive name fragment."""

    db = SessionLocal()
    try:
        query = db.query(Player)
        if search:
            query = query.filter(
                func.lower(Player.name).contains(search.lower())
            )
        players = query.order_by(Player.name).limit(50).all()
        return [{"id": p.player_id, "name": p.name} for p in players]
    finally:
        db.close()

@app.get("/players/{player_name}/elo")
def get_player_elo(player_name: str):
    """Return the stored Elo snapshot for a single player."""

    db = SessionLocal()
    try:
        player = find_player(db, player_name)
        elo = db.query(EloRating).filter_by(player_id=player.player_id).first()
        return {
            "name": player.name,
            "overall_elo": round(elo.overall_elo, 1),
            "hard_elo": round(elo.hard_elo, 1),
            "clay_elo": round(elo.clay_elo, 1),
            "grass_elo": round(elo.grass_elo, 1)
        }
    finally:
        db.close()

@app.get("/h2h")
def head_to_head(player1: str, player2: str):
    """Summarize the historical head-to-head record between two players."""

    db = SessionLocal()
    try:
        p1 = find_player(db, player1)
        p2 = find_player(db, player2)
        wr, _, total = get_h2h(db, p1.player_id, p2.player_id)
        hard_wr, _, hard_total = get_h2h(db, p1.player_id, p2.player_id, 'Hard')
        clay_wr, _, clay_total = get_h2h(db, p1.player_id, p2.player_id, 'Clay')
        grass_wr, _, grass_total = get_h2h(db, p1.player_id, p2.player_id, 'Grass')
        return {
            "player1": p1.name,
            "player2": p2.name,
            "overall": {"p1_win_rate": round(wr, 3), "total_matches": total},
            "hard": {"p1_win_rate": round(hard_wr, 3), "total_matches": hard_total},
            "clay": {"p1_win_rate": round(clay_wr, 3), "total_matches": clay_total},
            "grass": {"p1_win_rate": round(grass_wr, 3), "total_matches": grass_total}
        }
    finally:
        db.close()

@app.post("/predict", response_model=MatchResponse)
def predict_match(request: MatchRequest):
    """Assemble features for an upcoming match and score it with the model."""

    db = SessionLocal()
    try:
        # Resolve incoming names to the player ids used throughout the database.
        p1 = find_player(db, request.player1_name)
        p2 = find_player(db, request.player2_name)

        surf = request.surface

        # Elo snapshots are precomputed by the loader and stored separately from
        # the raw match history for quick lookup.
        p1_elo, p1_surf_elo = get_elo(db, p1.player_id, surf)
        p2_elo, p2_surf_elo = get_elo(db, p2.player_id, surf)

        # Recent form is calculated from the latest stored matches rather than
        # being saved as a separate table.
        p1_wr, p1_n = get_recent_wr(db, p1.player_id)
        p2_wr, p2_n = get_recent_wr(db, p2.player_id)

        # Surface-specific form uses the same idea, just filtered to the
        # requested surface.
        p1_surf_wr, p1_surf_n = get_recent_wr(db, p1.player_id, surf)
        p2_surf_wr, p2_surf_n = get_recent_wr(db, p2.player_id, surf)

        # Head-to-head features capture both the overall matchup history and the
        # matchup history on the requested surface.
        p1_h2h, p2_h2h, h2h_n = get_h2h(db, p1.player_id, p2.player_id)
        p1_h2h_s, p2_h2h_s, h2h_s_n = get_h2h(
            db, p1.player_id, p2.player_id, surf
        )

        # Days of rest are derived from each player's most recent recorded match.
        p1_rest = get_days_rest(db, p1.player_id)
        p2_rest = get_days_rest(db, p2.player_id)

        # The historical CSVs store rank values on each match row, so the API
        # uses the latest available rank from each player's most recent match.
        last_p1 = db.query(Match).filter(
            (Match.winner_id == p1.player_id) |
            (Match.loser_id == p1.player_id)
        ).order_by(Match.tourney_date.desc()).first()

        last_p2 = db.query(Match).filter(
            (Match.winner_id == p2.player_id) |
            (Match.loser_id == p2.player_id)
        ).order_by(Match.tourney_date.desc()).first()

        def get_rank(match, pid):
            if not match:
                return 100
            if match.winner_id == pid:
                return match.winner_rank
            return match.loser_rank

        p1_rank = get_rank(last_p1, p1.player_id)
        p2_rank = get_rank(last_p2, p2.player_id)

        # Tournament metadata is encoded numerically to match the training set.
        surf_enc = SURFACE_MAP.get(surf, 0)
        level_enc = LEVEL_MAP.get(request.tournament_level, 0)
        round_enc = ROUND_MAP.get(request.round, 3)

        p1_age = p1.age if p1.age else 27.0
        p2_age = p2.age if p2.age else 27.0

        # This dict mirrors `model/feature_cols.json`. Some features still use
        # neutral placeholders to match the current project behavior:
        # - `age_diff` stays fixed at zero
        # - rolling 7-day and 30-day match counts are not yet derived
        features = {
            'p1_rank': p1_rank,
            'p2_rank': p2_rank,
            'rank_diff': p2_rank - p1_rank,
            'p1_rank_points': p1_elo,
            'p2_rank_points': p2_elo,
            'rank_points_diff': p1_elo - p2_elo,
            'p1_age': p1_age,
            'p2_age': p2_age,
            'age_diff': 0.0,
            'surface': surf_enc,
            'level': level_enc,
            'indoor': 1 if request.indoor else 0,
            'round': round_enc,
            'best_of': request.best_of,
            'p1_recent_wr': p1_wr,
            'p2_recent_wr': p2_wr,
            'recent_wr_diff': p1_wr - p2_wr,
            'p1_recent_n': p1_n,
            'p2_recent_n': p2_n,
            'p1_surface_wr': p1_surf_wr,
            'p2_surface_wr': p2_surf_wr,
            'surface_wr_diff': p1_surf_wr - p2_surf_wr,
            'p1_surface_n': p1_surf_n,
            'p2_surface_n': p2_surf_n,
            'p1_h2h_wr': p1_h2h,
            'p2_h2h_wr': p2_h2h,
            'h2h_diff': p1_h2h - p2_h2h,
            'p1_h2h_n': h2h_n,
            'p1_h2h_surface_wr': p1_h2h_s,
            'p2_h2h_surface_wr': p2_h2h_s,
            'h2h_surface_diff': p1_h2h_s - p2_h2h_s,
            'p1_h2h_surface_n': h2h_s_n,
            'p1_days_rest': p1_rest,
            'p2_days_rest': p2_rest,
            'rest_diff': p1_rest - p2_rest,
            'p1_matches_7d': 0,
            'p2_matches_7d': 0,
            'matches_7d_diff': 0,
            'p1_matches_30d': 0,
            'p2_matches_30d': 0,
            'matches_30d_diff': 0,
            'p1_elo': p1_elo,
            'p2_elo': p2_elo,
            'elo_diff': p1_elo - p2_elo,
            'p1_surface_elo': p1_surf_elo,
            'p2_surface_elo': p2_surf_elo,
            'surface_elo_diff': p1_surf_elo - p2_surf_elo,
        }

        result = predict(features)

        # The model returns "player1" / "player2". The API upgrades that into
        # the resolved player names to make the response more readable.
        return MatchResponse(
            player1_win_probability=result['p1_win_probability'],
            player2_win_probability=result['p2_win_probability'],
            predicted_winner=p1.name if result['predicted_winner'] == 'player1' else p2.name,
            confidence=result['confidence'],
            player1_elo=round(p1_elo, 1),
            player2_elo=round(p2_elo, 1)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
