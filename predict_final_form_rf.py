from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).resolve().parent / "data" / "dataset_cleaned.csv"

CATEGORICAL_FEATURES = ["season", "home_team", "away_team"]
NUMERIC_FEATURES = [
    "round_number",
    # Away-form features for the current season
    "home_current_away_games_played",
    "home_current_away_win_rate",
    "home_current_away_draw_rate",
    "home_current_away_loss_rate",
    "home_current_away_goals_for_per_game",
    "home_current_away_goals_against_per_game",
    "home_current_away_goal_diff_per_game",
    "home_current_away_points_per_game",
    "away_current_away_games_played",
    "away_current_away_win_rate",
    "away_current_away_draw_rate",
    "away_current_away_loss_rate",
    "away_current_away_goals_for_per_game",
    "away_current_away_goals_against_per_game",
    "away_current_away_goal_diff_per_game",
    "away_current_away_points_per_game",
    # Away-form features from the last four completed seasons
    "home_history_away_games_played",
    "home_history_away_win_rate",
    "home_history_away_draw_rate",
    "home_history_away_loss_rate",
    "home_history_away_goals_for_per_game",
    "home_history_away_goals_against_per_game",
    "home_history_away_goal_diff_per_game",
    "home_history_away_points_per_game",
    "away_history_away_games_played",
    "away_history_away_win_rate",
    "away_history_away_draw_rate",
    "away_history_away_loss_rate",
    "away_history_away_goals_for_per_game",
    "away_history_away_goals_against_per_game",
    "away_history_away_goal_diff_per_game",
    "away_history_away_points_per_game",
    # Difference features
    "current_away_win_rate_diff",
    "current_away_goal_diff_per_game_diff",
    "current_away_points_per_game_diff",
    "history_away_win_rate_diff",
    "history_away_goal_diff_per_game_diff",
    "history_away_points_per_game_diff",
]


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    for column in ["match_number", "round_number", "home_goals", "away_goals", "goal_difference_home"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def empty_state() -> dict[str, float]:
    return {
        "games": 0.0,
        "wins": 0.0,
        "draws": 0.0,
        "losses": 0.0,
        "goals_for": 0.0,
        "goals_against": 0.0,
        "points": 0.0,
    }


def derive_features(state: dict[str, float], prefix: str) -> dict[str, float]:
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


def sum_states(states: list[dict[str, float]]) -> dict[str, float]:
    total = empty_state()
    for state in states:
        for key in total:
            total[key] += state[key]
    return total


def update_state(state: dict[str, float], goals_for: float, goals_against: float, result: str) -> None:
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


def build_pre_match_features(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.sort_values(["season", "match_number"]).reset_index()

    # Track away-form stats within the current season only.
    season_away_states: dict[tuple[str, str], dict[str, float]] = defaultdict(empty_state)

    # Track the last four completed seasons of away-form history for each team.
    away_history_by_team: dict[str, list[dict[str, float]]] = defaultdict(list)
    
    feature_rows = []
    current_season = None
    teams_in_season: set[str] = set()

    for idx, row in ordered.iterrows():
        season = row["season"]
        home_team = row["home_team"]
        away_team = row["away_team"]

        # When moving to a new season, snapshot the completed season's away form into history.
        if current_season != season and current_season is not None:
            for team in teams_in_season:
                snapshot = season_away_states[(current_season, team)]
                if snapshot["games"] > 0:
                    away_history_by_team[team].append(snapshot.copy())
                    away_history_by_team[team] = away_history_by_team[team][-4:]
            
            teams_in_season = set()
        
        current_season = season
        teams_in_season.add(home_team)
        teams_in_season.add(away_team)

        # Current season away form and prior four-season away history.
        home_current_away_state = season_away_states[(season, home_team)]
        away_current_away_state = season_away_states[(season, away_team)]
        home_history_away_state = sum_states(away_history_by_team[home_team])
        away_history_away_state = sum_states(away_history_by_team[away_team])

        home_current_away_features = derive_features(home_current_away_state, "home_current_away")
        away_current_away_features = derive_features(away_current_away_state, "away_current_away")
        home_history_away_features = derive_features(home_history_away_state, "home_history_away")
        away_history_away_features = derive_features(away_history_away_state, "away_history_away")

        feature_rows.append(
            {
                "row_index": row["index"],
                "season": season,
                "round_number": row["round_number"],
                "home_team": home_team,
                "away_team": away_team,
                **home_current_away_features,
                **away_current_away_features,
                **home_history_away_features,
                **away_history_away_features,
                "current_away_win_rate_diff": home_current_away_features["home_current_away_win_rate"] - away_current_away_features["away_current_away_win_rate"],
                "current_away_goal_diff_per_game_diff": home_current_away_features["home_current_away_goal_diff_per_game"] - away_current_away_features["away_current_away_goal_diff_per_game"],
                "current_away_points_per_game_diff": home_current_away_features["home_current_away_points_per_game"] - away_current_away_features["away_current_away_points_per_game"],
                "history_away_win_rate_diff": home_history_away_features["home_history_away_win_rate"] - away_history_away_features["away_history_away_win_rate"],
                "history_away_goal_diff_per_game_diff": home_history_away_features["home_history_away_goal_diff_per_game"] - away_history_away_features["away_history_away_goal_diff_per_game"],
                "history_away_points_per_game_diff": home_history_away_features["home_history_away_points_per_game"] - away_history_away_features["away_history_away_points_per_game"],
            }
        )

        if pd.notna(row["home_goals"]) and pd.notna(row["away_goals"]):
            away_result = "A" if row["result_90min"] == "H" else "H" if row["result_90min"] == "A" else "D"
            update_state(
                season_away_states[(season, away_team)],
                row["away_goals"],
                row["home_goals"],
                away_result,
            )

    return pd.DataFrame(feature_rows).set_index("row_index")


def build_model() -> GridSearchCV:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    param_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [8, 16],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"],
    }

    return GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=1,
    )


def main() -> None:
    data = load_data()
    features = build_pre_match_features(data)
    model_data = data.join(features.drop(columns=CATEGORICAL_FEATURES + ["round_number"]))

    train = model_data[
        model_data["knockout_winner"].notna()
        & model_data["home_goals"].notna()
        & model_data["away_goals"].notna()
    ].copy()

    train = train[
        (train["knockout_winner"] == train["home_team"])
        | (train["knockout_winner"] == train["away_team"])
    ]
    train["home_win_target"] = (train["knockout_winner"] == train["home_team"]).astype(int)

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X_train = train[feature_columns]
    y_train = train["home_win_target"]

    class_counts = pd.Series(y_train).value_counts()
    n_splits = min(5, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("Not enough knockout results to train the model.")

    model = build_model()
    model.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    model.fit(X_train, y_train)

    print("Best model settings:", model.best_params_)
    print(f"Cross-validation accuracy: {model.best_score_:.4f}")

    final_mask = (
        model_data["season"].eq("2025-26")
        & model_data["home_team"].eq("Paris")
        & model_data["away_team"].eq("Arsenal")
        & model_data["round_number"].eq(17)
    )
    final_match = model_data.loc[final_mask, feature_columns]

    if final_match.empty:
        raise ValueError("Could not find the 2026 final row for Paris vs Arsenal.")

    home_win_prediction = int(model.predict(final_match)[0])
    prediction = "Paris" if home_win_prediction == 1 else "Arsenal"
    probabilities = model.predict_proba(final_match)[0]

    print("\nPredicted Champions League 2026 final winner:")
    print(prediction)
    print("\nBinary probabilities:")
    print(f"Paris : {probabilities[1]:.4f}")
    print(f"Arsenal : {probabilities[0]:.4f}")


if __name__ == "__main__":
    main()
