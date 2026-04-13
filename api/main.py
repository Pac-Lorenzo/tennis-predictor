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
from datetime import date, timedelta

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
    player1_name: str
    player2_name: str
    surface: str
    tournament_level: str
    round: str
    best_of: int
    indoor: bool = False

class MatchResponse(BaseModel):
    player1_win_probability: float
    player2_win_probability: float
    predicted_winner: str
    confidence: str
    player1_elo: float
    player2_elo: float

# ── Encoding maps ─────────────────────────────────────────────
SURFACE_MAP = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 0}
LEVEL_MAP   = {'250': 0, '500': 1, 'A': 2, 'M': 3, 'G': 4, 'D': 5, 'O': 5, 'F': 6}
ROUND_MAP   = {'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4, 'QF': 5, 'SF': 6, 'F': 7, 'RR': 3, 'BR': 6}

# ── Helper functions ──────────────────────────────────────────
def find_player(db, name: str):
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
    elo = db.query(EloRating).filter_by(player_id=player_id).first()
    if not elo:
        return 1500.0, 1500.0
    surface_elo_map = {
        'Hard':   elo.hard_elo,
        'Clay':   elo.clay_elo,
        'Grass':  elo.grass_elo,
        'Carpet': elo.hard_elo,
    }
    return elo.overall_elo, surface_elo_map.get(surface, elo.overall_elo)

def get_recent_wr(db, player_id: str, surface: str = None, n: int = 10):
    """Win rate over last n matches, with Laplace smoothing to match training."""
    matches = db.query(Match).filter(
        (Match.winner_id == player_id) | (Match.loser_id == player_id)
    )
    if surface:
        matches = matches.filter(Match.surface == surface)
    matches = matches.order_by(Match.tourney_date.desc()).limit(n).all()

    if not matches:
        return (2.5 / 5.0), 0   # smoothed prior = 0.5 with 0 matches

    wins = sum(1 for m in matches if m.winner_id == player_id)
    n_actual = len(matches)
    # Apply same Laplace smoothing as training: (wins+2.5)/(n+5)
    smoothed = (wins + 2.5) / (n_actual + 5)
    return smoothed, n_actual

def get_h2h(db, p1_id: str, p2_id: str, surface: str = None):
    """H2H win rates with Laplace smoothing to match training."""
    matches = db.query(Match).filter(
        ((Match.winner_id == p1_id) & (Match.loser_id == p2_id)) |
        ((Match.winner_id == p2_id) & (Match.loser_id == p1_id))
    )
    if surface:
        matches = matches.filter(Match.surface == surface)
    matches = matches.all()

    total = len(matches)
    if total == 0:
        return 0.5, 0.5, 0

    p1_wins = sum(1 for m in matches if m.winner_id == p1_id)
    # Laplace smoothing: (wins+1)/(total+2) — matches training pipeline
    p1_wr = (p1_wins + 1) / (total + 2)
    return p1_wr, 1 - p1_wr, total

def get_days_rest(db, player_id: str):
    last = db.query(Match).filter(
        (Match.winner_id == player_id) | (Match.loser_id == player_id)
    ).order_by(Match.tourney_date.desc()).first()
    if not last:
        return 30
    delta = (date.today() - last.tourney_date).days
    return min(delta, 60)   # capped at 60 to match training

def get_rank_and_points(match, player_id: str):
    """Extract rank and rank points from a player's most recent match row."""
    if not match:
        return 100, 0
    if match.winner_id == player_id:
        rank = match.winner_rank or 100
        pts  = match.winner_rank_points or 0
    else:
        rank = match.loser_rank or 100
        pts  = match.loser_rank_points or 0

    return rank, pts

def get_match_counts(db, player_id: str):
    """Count matches played in last 7 and 30 days."""
    today    = date.today()
    date_7d  = today - timedelta(days=7)
    date_30d = today - timedelta(days=30)

    base = db.query(Match).filter(
        (Match.winner_id == player_id) | (Match.loser_id == player_id)
    )
    count_7d  = base.filter(Match.tourney_date >= date_7d).count()
    count_30d = base.filter(Match.tourney_date >= date_30d).count()
    return count_7d, count_30d

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/debug/{player_name}")
def debug_player(player_name: str, surface: str = "Clay"):
    db = SessionLocal()
    try:
        player  = find_player(db, player_name)
        wr, n   = get_recent_wr(db, player.player_id)
        surf_wr, surf_n = get_recent_wr(db, player.player_id, surface)
        rest    = get_days_rest(db, player.player_id)
        elo_overall, elo_surf = get_elo(db, player.player_id, surface)
        last    = db.query(Match).filter(
            (Match.winner_id == player.player_id) | (Match.loser_id == player.player_id)
        ).order_by(Match.tourney_date.desc()).first()
        _, pts  = get_rank_and_points(last, player.player_id)
        c7, c30 = get_match_counts(db, player.player_id)
        return {
            "name":            player.name,
            "recent_wr":       round(wr, 3),
            "recent_n":        n,
            f"{surface}_wr":   round(surf_wr, 3),
            f"{surface}_n":    surf_n,
            "days_rest":       rest,
            "matches_7d":      c7,
            "matches_30d":     c30,
            "rank_points":     pts,
            "overall_elo":     round(elo_overall, 1),
            f"{surface}_elo":  round(elo_surf, 1),
        }
    finally:
        db.close()

@app.get("/players")
def list_players(search: str = ""):
    db = SessionLocal()
    try:
        query = db.query(Player)
        if search:
            query = query.filter(func.lower(Player.name).contains(search.lower()))
        players = query.order_by(Player.name).limit(50).all()
        return [{"id": p.player_id, "name": p.name} for p in players]
    finally:
        db.close()

@app.get("/players/{player_name}/elo")
def get_player_elo(player_name: str):
    db = SessionLocal()
    try:
        player = find_player(db, player_name)
        elo    = db.query(EloRating).filter_by(player_id=player.player_id).first()
        return {
            "name":        player.name,
            "overall_elo": round(elo.overall_elo, 1),
            "hard_elo":    round(elo.hard_elo, 1),
            "clay_elo":    round(elo.clay_elo, 1),
            "grass_elo":   round(elo.grass_elo, 1),
        }
    finally:
        db.close()

@app.get("/h2h")
def head_to_head(player1: str, player2: str):
    db = SessionLocal()
    try:
        p1 = find_player(db, player1)
        p2 = find_player(db, player2)
        wr,       _, total       = get_h2h(db, p1.player_id, p2.player_id)
        hard_wr,  _, hard_total  = get_h2h(db, p1.player_id, p2.player_id, 'Hard')
        clay_wr,  _, clay_total  = get_h2h(db, p1.player_id, p2.player_id, 'Clay')
        grass_wr, _, grass_total = get_h2h(db, p1.player_id, p2.player_id, 'Grass')
        return {
            "player1": p1.name, "player2": p2.name,
            "overall": {"p1_win_rate": round(wr, 3),       "total_matches": total},
            "hard":    {"p1_win_rate": round(hard_wr, 3),  "total_matches": hard_total},
            "clay":    {"p1_win_rate": round(clay_wr, 3),  "total_matches": clay_total},
            "grass":   {"p1_win_rate": round(grass_wr, 3), "total_matches": grass_total},
        }
    finally:
        db.close()

@app.post("/predict", response_model=MatchResponse)
def predict_match(request: MatchRequest):
    db = SessionLocal()
    try:
        p1   = find_player(db, request.player1_name)
        p2   = find_player(db, request.player2_name)
        surf = request.surface

        # ── ELO ──────────────────────────────────────────────
        p1_elo,      p1_surf_elo  = get_elo(db, p1.player_id, surf)
        p2_elo,      p2_surf_elo  = get_elo(db, p2.player_id, surf)

        # ── Recent form ───────────────────────────────────────
        p1_wr,      p1_n         = get_recent_wr(db, p1.player_id)
        p2_wr,      p2_n         = get_recent_wr(db, p2.player_id)
        p1_surf_wr, p1_surf_n    = get_recent_wr(db, p1.player_id, surf)
        p2_surf_wr, p2_surf_n    = get_recent_wr(db, p2.player_id, surf)

        # ── H2H ──────────────────────────────────────────────
        p1_h2h,   p2_h2h,   h2h_n   = get_h2h(db, p1.player_id, p2.player_id)
        p1_h2h_s, p2_h2h_s, h2h_s_n = get_h2h(db, p1.player_id, p2.player_id, surf)

        # ── Rest ──────────────────────────────────────────────
        p1_rest = get_days_rest(db, p1.player_id)
        p2_rest = get_days_rest(db, p2.player_id)

        # ── Match counts (7d / 30d) ───────────────────────────
        p1_7d, p1_30d = get_match_counts(db, p1.player_id)
        p2_7d, p2_30d = get_match_counts(db, p2.player_id)

        # ── Rank + rank points (from last match row) ──────────
        last_p1 = db.query(Match).filter(
            (Match.winner_id == p1.player_id) | (Match.loser_id == p1.player_id)
        ).order_by(Match.tourney_date.desc()).first()

        last_p2 = db.query(Match).filter(
            (Match.winner_id == p2.player_id) | (Match.loser_id == p2.player_id)
        ).order_by(Match.tourney_date.desc()).first()

        p1_rank, p1_rank_pts = get_rank_and_points(last_p1, p1.player_id)
        p2_rank, p2_rank_pts = get_rank_and_points(last_p2, p2.player_id)

        # ── Age ───────────────────────────────────────────────
        p1_age = float(p1.age) if p1.age else 27.0
        p2_age = float(p2.age) if p2.age else 27.0

        # ── Encodings ─────────────────────────────────────────
        surf_enc  = SURFACE_MAP.get(surf, 0)
        level_enc = LEVEL_MAP.get(request.tournament_level, 0)
        round_enc = ROUND_MAP.get(request.round, 3)

        # ── Feature vector ────────────────────────────────────
        features = {
            'p1_rank':             p1_rank,
            'p2_rank':             p2_rank,
            'rank_diff':           p2_rank - p1_rank,
            'p1_rank_points':      p1_rank_pts,  # rank_points not stored in DB — neutral placeholder
            'p2_rank_points':      p2_rank_pts,
            'rank_points_diff':    p1_rank_pts - p2_rank_pts,
            'p1_age':              p1_age,
            'p2_age':              p2_age,
            'age_diff':            p1_age - p2_age,
            'surface':             surf_enc,
            'level':               level_enc,
            'indoor':              1 if request.indoor else 0,
            'round':               round_enc,
            'best_of':             request.best_of,
            'p1_recent_wr':        p1_wr,
            'p2_recent_wr':        p2_wr,
            'recent_wr_diff':      p1_wr - p2_wr,
            'p1_recent_n':         p1_n,
            'p2_recent_n':         p2_n,
            'p1_surface_wr':       p1_surf_wr,
            'p2_surface_wr':       p2_surf_wr,
            'surface_wr_diff':     p1_surf_wr - p2_surf_wr,
            'p1_surface_n':        p1_surf_n,
            'p2_surface_n':        p2_surf_n,
            'p1_h2h_wr':           p1_h2h,
            'p2_h2h_wr':           p2_h2h,
            'h2h_diff':            p1_h2h - p2_h2h,
            'p1_h2h_n':            h2h_n,
            'p1_h2h_surface_wr':   p1_h2h_s,
            'p2_h2h_surface_wr':   p2_h2h_s,
            'h2h_surface_diff':    p1_h2h_s - p2_h2h_s,
            'p1_h2h_surface_n':    h2h_s_n,
            'p1_days_rest':        p1_rest,
            'p2_days_rest':        p2_rest,
            'rest_diff':           p1_rest - p2_rest,
            'p1_matches_7d':       p1_7d,
            'p2_matches_7d':       p2_7d,
            'matches_7d_diff':     p1_7d - p2_7d,
            'p1_matches_30d':      p1_30d,
            'p2_matches_30d':      p2_30d,
            'matches_30d_diff':    p1_30d - p2_30d,
            'p1_elo':              p1_elo,
            'p2_elo':              p2_elo,
            'elo_diff':            p1_elo - p2_elo,
            'p1_surface_elo':      p1_surf_elo,
            'p2_surface_elo':      p2_surf_elo,
            'surface_elo_diff':    p1_surf_elo - p2_surf_elo,
        }

        result = predict(features)

        return MatchResponse(
            player1_win_probability=result['p1_win_probability'],
            player2_win_probability=result['p2_win_probability'],
            predicted_winner=p1.name if result['predicted_winner'] == 'player1' else p2.name,
            confidence=result['confidence'],
            player1_elo=round(p1_elo, 1),
            player2_elo=round(p2_elo, 1),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
