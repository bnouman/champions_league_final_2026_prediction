from collections import defaultdict
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "dataset_cleaned.csv"

def empty_state():
    return {
        "games": 0.0,
        "wins": 0.0,
        "draws": 0.0,
        "losses": 0.0,
        "goals_for": 0.0,
        "goals_against": 0.0,
        "points": 0.0,
    }

def update_state(state, goals_for, goals_against, result):
    state["games"] += 1
    state["goals_for"] += goals_for
    state["goals_against"] += goals_against

    if result == "H":
        state["wins"] += 1
        state["points"] += 3
    elif result == "A":
        state["losses"] += 1
    else:
        state["draws"] += 1
        state["points"] += 1

def derive_features(state, prefix):
    games = state["games"]
    win_rate = state["wins"] / games if games else 0.0
    draw_rate = state["draws"] / games if games else 0.0
    loss_rate = state["losses"] / games if games else 0.0
    goals_for_per_game = state["goals_for"] / games if games else 0.0
    goals_against_per_game = state["goals_against"] / games if games else 0.0
    goal_diff_per_game = (state["goals_for"] - state["goals_against"]) / games if games else 0.0
    points_per_game = state["points"] / games if games else 0.0

    return {
        f"{prefix}_games_played": games,
        f"{prefix}_win_rate": win_rate,
        f"{prefix}_draw_rate": draw_rate,
        f"{prefix}_loss_rate": loss_rate,
        f"{prefix}_goals_for_per_game": goals_for_per_game,
        f"{prefix}_goals_against_per_game": goals_against_per_game,
        f"{prefix}_goal_diff_per_game": goal_diff_per_game,
        f"{prefix}_points_per_game": points_per_game,
    }

def derive_career_features(home_state, away_state, team_name, prefix):
    """Derive career features from home and away specific states."""
    home_games = home_state["games"]
    away_games = away_state["games"]
    
    home_win_rate = home_state["wins"] / home_games if home_games else 0.0
    away_win_rate = away_state["wins"] / away_games if away_games else 0.0
    home_goals_per_game = home_state["goals_for"] / home_games if home_games else 0.0
    away_goals_per_game = away_state["goals_for"] / away_games if away_games else 0.0
    home_goal_diff = (home_state["goals_for"] - home_state["goals_against"]) / home_games if home_games else 0.0
    away_goal_diff = (away_state["goals_for"] - away_state["goals_against"]) / away_games if away_games else 0.0
    
    return {
        f"{prefix}_career_home_win_rate": home_win_rate,
        f"{prefix}_career_away_win_rate": away_win_rate,
        f"{prefix}_career_home_goals_per_game": home_goals_per_game,
        f"{prefix}_career_away_goals_per_game": away_goals_per_game,
        f"{prefix}_career_home_goal_diff": home_goal_diff,
        f"{prefix}_career_away_goal_diff": away_goal_diff,
    }

def build_pre_match_features(data):
    print("Starting feature engineering...")
    ordered = data.sort_values(["season", "match_number"]).reset_index()
    print(f"Sorted {len(ordered)} rows by season and match_number")
    
    # Track season-specific stats: {(season, team, location): state}
    season_states = defaultdict(empty_state)
    
    # Track career stats: {team: {"home": state, "away": state}}
    career_states = defaultdict(lambda: {"home": empty_state(), "away": empty_state()})
    
    feature_rows = []
    current_season = None
    teams_in_season = set()
    match_count = 0

    for idx, row in ordered.iterrows():
        match_count += 1
        if match_count % 100 == 0:
            print(f"Processing match {match_count}/{len(ordered)}")
        
        season = row["season"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        # When moving to a new season, accumulate prior season's stats into career for teams in that season
        if current_season != season and current_season is not None:
            print(f"  Season change: {current_season} -> {season}, accumulating {len(teams_in_season)} teams")
            # Add prior season's stats to career for all teams that played in that season
            for team in teams_in_season:
                season_home = season_states[(current_season, team, "home")]
                season_away = season_states[(current_season, team, "away")]
                
                if season_home["games"] > 0:
                    career = career_states[team]["home"]
                    career["games"] += season_home["games"]
                    career["wins"] += season_home["wins"]
                    career["draws"] += season_home["draws"]
                    career["losses"] += season_home["losses"]
                    career["goals_for"] += season_home["goals_for"]
                    career["goals_against"] += season_home["goals_against"]
                    career["points"] += season_home["points"]
                
                if season_away["games"] > 0:
                    career = career_states[team]["away"]
                    career["games"] += season_away["games"]
                    career["wins"] += season_away["wins"]
                    career["draws"] += season_away["draws"]
                    career["losses"] += season_away["losses"]
                    career["goals_for"] += season_away["goals_for"]
                    career["goals_against"] += season_away["goals_against"]
                    career["points"] += season_away["points"]
            
            teams_in_season = set()
        
        current_season = season
        teams_in_season.add(home_team)
        teams_in_season.add(away_team)

        # Current season stats
        home_season_state = season_states[(season, home_team, "home")]
        away_season_state = season_states[(season, away_team, "away")]

        home_features = derive_features(home_season_state, "home")
        away_features = derive_features(away_season_state, "away")
        
        # Career stats (all prior seasons)
        home_career_features = derive_career_features(
            career_states[home_team]["home"],
            career_states[home_team]["away"],
            home_team,
            "home"
        )
        
        away_career_features = derive_career_features(
            career_states[away_team]["home"],
            career_states[away_team]["away"],
            away_team,
            "away"
        )

        feature_rows.append(
            {
                "row_index": row["index"],
                "season": season,
                "round_number": row["round_number"],
                "home_team": home_team,
                "away_team": away_team,
                **home_features,
                **away_features,
                **home_career_features,
                **away_career_features,
                "win_rate_diff": home_features["home_win_rate"] - away_features["away_win_rate"],
                "goal_diff_per_game_diff": home_features["home_goal_diff_per_game"] - away_features["away_goal_diff_per_game"],
                "points_per_game_diff": home_features["home_points_per_game"] - away_features["away_points_per_game"],
                "career_home_win_rate_diff": home_career_features["home_career_home_win_rate"] - away_career_features["away_career_home_win_rate"],
                "career_away_win_rate_diff": home_career_features["home_career_away_win_rate"] - away_career_features["away_career_away_win_rate"],
            }
        )

        if pd.notna(row["home_goals"]) and pd.notna(row["away_goals"]):
            # Update season-level home team's home stats
            update_state(home_season_state, row["home_goals"], row["away_goals"], row["result_90min"])
            
            # Update season-level away team's away stats
            away_result = "A" if row["result_90min"] == "H" else "H" if row["result_90min"] == "A" else "D"
            update_state(away_season_state, row["away_goals"], row["home_goals"], away_result)

    print(f"Created {len(feature_rows)} feature rows")
    return pd.DataFrame(feature_rows).set_index("row_index")

def main():
    print("Loading data...")
    data = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(data)} rows")
    
    print("Building features...")
    features = build_pre_match_features(data)
    print(f"Feature engineering complete: {len(features)} rows")
    print(features.head())

if __name__ == "__main__":
    main()
