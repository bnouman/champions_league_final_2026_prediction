from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).resolve().parent / "data" / "dataset_cleaned.csv"

FEATURE_COLUMNS = [
    "round_number",
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
    "current_away_win_rate_diff",
    "current_away_goal_diff_per_game_diff",
    "current_away_points_per_game_diff",
    "history_away_win_rate_diff",
    "history_away_goal_diff_per_game_diff",
    "history_away_points_per_game_diff",
]

NUMERIC_COLUMNS = FEATURE_COLUMNS
RF_CATEGORICAL_COLUMNS = ["season", "home_team", "away_team"]
RF_NUMERIC_COLUMNS = FEATURE_COLUMNS


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


def build_pre_match_features(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.sort_values(["season", "match_number"]).reset_index()

    season_away_states: dict[tuple[str, str], dict[str, float]] = defaultdict(empty_state)
    away_history_by_team: dict[str, list[dict[str, float]]] = defaultdict(list)

    feature_rows = []
    current_season = None
    teams_in_season: set[str] = set()

    for _, row in ordered.iterrows():
        season = row["season"]
        home_team = row["home_team"]
        away_team = row["away_team"]

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


def standardize_frame(frame: pd.DataFrame, mean: pd.Series | None = None, std: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if mean is None:
        mean = frame.mean()
    if std is None:
        std = frame.std(ddof=0).replace(0.0, 1.0)
    standardized = (frame - mean) / std
    return standardized, mean, std


def make_indices(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    season_codes = frame["season"].astype("category")
    home_team_codes = frame["home_team"].astype("category")
    away_team_codes = frame["away_team"].astype("category")
    return season_codes, home_team_codes, away_team_codes


def build_training_frame(data: pd.DataFrame) -> pd.DataFrame:
    features = build_pre_match_features(data)
    model_data = data.join(features.drop(columns=["season", "round_number", "home_team", "away_team"]))

    train = model_data[
        model_data["knockout_winner"].notna()
        & model_data["home_goals"].notna()
        & model_data["away_goals"].notna()
    ].copy()

    train = train[
        (train["knockout_winner"] == train["home_team"])
        | (train["knockout_winner"] == train["away_team"])
    ].copy()
    train["home_win_target"] = (train["knockout_winner"] == train["home_team"]).astype(int)
    return train


def choose_validation_season(train: pd.DataFrame, final_season: str = "2025-26") -> str:
    seasons = sorted(train["season"].astype(str).unique())
    non_final_seasons = [season for season in seasons if season != final_season]
    if not non_final_seasons:
        return seasons[-1]
    return non_final_seasons[-1]


def split_temporal_holdout(train: pd.DataFrame, validation_season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_frame = train[train["season"] < validation_season].copy()
    validation_frame = train[train["season"] == validation_season].copy()
    return fit_frame, validation_frame


def build_rf_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                RF_CATEGORICAL_COLUMNS,
            ),
            ("num", "passthrough", RF_NUMERIC_COLUMNS),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_rf_temporal(fit_frame: pd.DataFrame, validation_frame: pd.DataFrame) -> dict[str, float]:
    if fit_frame.empty or validation_frame.empty:
        return {"accuracy": float("nan"), "brier": float("nan")}

    feature_columns = RF_CATEGORICAL_COLUMNS + RF_NUMERIC_COLUMNS
    model = build_rf_pipeline()
    model.fit(fit_frame[feature_columns], fit_frame["home_win_target"])

    probabilities = model.predict_proba(validation_frame[feature_columns])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    truth = validation_frame["home_win_target"].to_numpy(dtype=int)

    accuracy = float((predictions == truth).mean())
    brier = float(np.mean((probabilities - truth) ** 2))
    return {"accuracy": accuracy, "brier": brier}


def fit_hierarchical_model(train: pd.DataFrame) -> tuple[pm.Model, object, pd.Series, pd.Series, dict[str, int], dict[str, int], dict[str, int]]:
    numeric_frame = train[NUMERIC_COLUMNS].copy()
    numeric_frame["round_number"] = pd.to_numeric(numeric_frame["round_number"], errors="coerce")
    numeric_frame = numeric_frame.fillna(0.0)
    numeric_scaled, mean, std = standardize_frame(numeric_frame)

    season_categories = sorted(train["season"].astype(str).unique())
    team_categories = sorted(set(train["home_team"].astype(str)).union(train["away_team"].astype(str)))
    season_to_idx = {season: idx for idx, season in enumerate(season_categories)}
    team_to_idx = {team: idx for idx, team in enumerate(team_categories)}

    season_idx = train["season"].map(season_to_idx).to_numpy(dtype=int)
    home_team_idx = train["home_team"].map(team_to_idx).to_numpy(dtype=int)
    away_team_idx = train["away_team"].map(team_to_idx).to_numpy(dtype=int)
    y = train["home_win_target"].to_numpy(dtype=int)
    X = numeric_scaled.to_numpy(dtype=float)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        beta = pm.Normal("beta", mu=0.0, sigma=0.75, shape=X.shape[1])

        sigma_team = pm.HalfNormal("sigma_team", sigma=1.0)
        z_team = pm.Normal("z_team", mu=0.0, sigma=1.0, shape=len(team_categories))
        team_strength = pm.Deterministic("team_strength", z_team * sigma_team)

        sigma_season = pm.HalfNormal("sigma_season", sigma=1.0)
        z_season = pm.Normal("z_season", mu=0.0, sigma=1.0, shape=len(season_categories))
        season_effect = pm.Deterministic("season_effect", z_season * sigma_season)

        logit_p = alpha + pm.math.dot(X, beta) + team_strength[home_team_idx] - team_strength[away_team_idx] + season_effect[season_idx]
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))

        pm.Bernoulli("obs", logit_p=logit_p, observed=y)

        trace = pm.sample(
            draws=300,
            tune=400,
            chains=2,
            cores=1,
            target_accept=0.99,
            random_seed=42,
            progressbar=False,
            return_inferencedata=False,
        )

    return model, trace, mean, std, season_to_idx, team_to_idx, {"n_features": X.shape[1]}


def posterior_mean_probability(
    trace: object,
    numeric_row: pd.Series,
    season: str,
    home_team: str,
    away_team: str,
    mean: pd.Series,
    std: pd.Series,
    season_to_idx: dict[str, int],
    team_to_idx: dict[str, int],
) -> float:
    row_values_series = pd.to_numeric(numeric_row[NUMERIC_COLUMNS], errors="coerce").fillna(0.0)
    row_frame = pd.DataFrame([row_values_series])
    row_scaled = (row_frame - mean) / std
    row_values = row_scaled.to_numpy(dtype=float).reshape(-1)

    alpha = np.asarray(trace.get_values("alpha", combine=True)).reshape(-1)
    beta = np.asarray(trace.get_values("beta", combine=True))
    team_strength = np.asarray(trace.get_values("team_strength", combine=True))
    season_effect = np.asarray(trace.get_values("season_effect", combine=True))

    if beta.shape[0] == row_values.shape[0]:
        beta_term = beta.T @ row_values
    else:
        beta_term = beta @ row_values

    home_idx = team_to_idx.get(home_team)
    away_idx = team_to_idx.get(away_team)
    if home_idx is None or away_idx is None:
        team_term = 0.0
    elif team_strength.shape[0] == len(team_to_idx):
        team_term = team_strength[home_idx] - team_strength[away_idx]
    else:
        team_term = team_strength[:, home_idx] - team_strength[:, away_idx]

    season_idx = season_to_idx.get(season)
    if season_idx is None:
        season_term = 0.0
    elif season_effect.ndim == 1:
        season_term = season_effect[season_idx]
    else:
        season_term = season_effect[:, season_idx]

    linear_term = alpha + beta_term + team_term + season_term
    probabilities = 1.0 / (1.0 + np.exp(-linear_term))
    return float(np.asarray(probabilities).mean())


def swap_match_perspective(row: pd.Series) -> pd.Series:
    swapped = row.copy()

    swapped["home_team"] = row["away_team"]
    swapped["away_team"] = row["home_team"]

    for column in list(row.index):
        if column.startswith("home_"):
            counterpart = "away_" + column[len("home_"):]
            if counterpart in row.index:
                swapped[column] = row[counterpart]
                swapped[counterpart] = row[column]

    for diff_column in [
        "current_away_win_rate_diff",
        "current_away_goal_diff_per_game_diff",
        "current_away_points_per_game_diff",
        "history_away_win_rate_diff",
        "history_away_goal_diff_per_game_diff",
        "history_away_points_per_game_diff",
    ]:
        if diff_column in row.index:
            swapped[diff_column] = -row[diff_column]

    return swapped


def neutral_venue_probability(
    trace: object,
    row: pd.Series,
    team_a: str,
    team_b: str,
    mean: pd.Series,
    std: pd.Series,
    season_to_idx: dict[str, int],
    team_to_idx: dict[str, int],
) -> float:
    # team_a as listed home, team_b as listed away
    team_a_as_home = posterior_mean_probability(
        trace=trace,
        numeric_row=row,
        season=str(row["season"]),
        home_team=team_a,
        away_team=team_b,
        mean=mean,
        std=std,
        season_to_idx=season_to_idx,
        team_to_idx=team_to_idx,
    )

    # Reverse perspective and convert back to probability of team_a winning.
    swapped = swap_match_perspective(row)
    team_b_as_home = posterior_mean_probability(
        trace=trace,
        numeric_row=swapped,
        season=str(swapped["season"]),
        home_team=team_b,
        away_team=team_a,
        mean=mean,
        std=std,
        season_to_idx=season_to_idx,
        team_to_idx=team_to_idx,
    )

    team_a_as_away = 1.0 - team_b_as_home
    return 0.5 * (team_a_as_home + team_a_as_away)


def evaluate_bayes_holdout(
    validation_frame: pd.DataFrame,
    mean: pd.Series,
    std: pd.Series,
    season_to_idx: dict[str, int],
    team_to_idx: dict[str, int],
    trace: object,
) -> dict[str, float]:
    if validation_frame.empty:
        return {"accuracy": float("nan"), "brier": float("nan")}

    probabilities = []
    predictions = []
    truth = validation_frame["home_win_target"].to_numpy(dtype=int)

    for _, row in validation_frame.iterrows():
        probability = posterior_mean_probability(
            trace=trace,
            numeric_row=row,
            season=row["season"],
            home_team=row["home_team"],
            away_team=row["away_team"],
            mean=mean,
            std=std,
            season_to_idx=season_to_idx,
            team_to_idx=team_to_idx,
        )
        probabilities.append(probability)
        predictions.append(int(probability >= 0.5))

    predictions_array = np.array(predictions, dtype=int)
    accuracy = float((predictions_array == truth).mean())
    brier = float(np.mean((np.array(probabilities) - truth) ** 2))
    return {"accuracy": accuracy, "brier": brier}


def main() -> None:
    data = load_data()
    train = build_training_frame(data)

    if train.empty:
        raise ValueError("No completed knockout matches available for Bayesian training.")

    validation_season = choose_validation_season(train, final_season="2025-26")
    fit_frame, validation_frame = split_temporal_holdout(train, validation_season)

    if fit_frame.empty:
        raise ValueError("Temporal split produced an empty training set. Add older completed seasons first.")
    if validation_frame.empty:
        raise ValueError("Temporal split produced an empty validation set.")

    _, eval_trace, eval_mean, eval_std, eval_season_to_idx, eval_team_to_idx, _ = fit_hierarchical_model(fit_frame)
    bayes_metrics = evaluate_bayes_holdout(
        validation_frame=validation_frame,
        mean=eval_mean,
        std=eval_std,
        season_to_idx=eval_season_to_idx,
        team_to_idx=eval_team_to_idx,
        trace=eval_trace,
    )
    rf_metrics = evaluate_rf_temporal(fit_frame, validation_frame)

    eval_divergences = int(np.asarray(eval_trace.get_sampler_stats("diverging", combine=True)).sum())

    print("Temporal holdout comparison (same split):")
    print(f"Validation season: {validation_season}")
    print(f"Random Forest -> accuracy: {rf_metrics['accuracy']:.4f}, brier: {rf_metrics['brier']:.4f}")
    print(f"Hierarchical Bayes -> accuracy: {bayes_metrics['accuracy']:.4f}, brier: {bayes_metrics['brier']:.4f}, divergences: {eval_divergences}")

    # Refit on all available completed knockout matches for the final prediction.
    model_fit_frame = train.copy()
    model, trace, mean, std, season_to_idx, team_to_idx, _ = fit_hierarchical_model(model_fit_frame)

    print("\nHierarchical Bayesian model refit on all completed data.")
    alpha_mean = float(np.asarray(trace.get_values("alpha", combine=True)).mean())
    sigma_team_mean = float(np.asarray(trace.get_values("sigma_team", combine=True)).mean())
    sigma_season_mean = float(np.asarray(trace.get_values("sigma_season", combine=True)).mean())
    train_divergences = int(np.asarray(trace.get_sampler_stats("diverging", combine=True)).sum())
    print(
        "Posterior means -> "
        f"alpha: {alpha_mean:.4f}, sigma_team: {sigma_team_mean:.4f}, sigma_season: {sigma_season_mean:.4f}, "
        f"divergences: {train_divergences}"
    )

    features = build_pre_match_features(data)
    model_data = data.join(features.drop(columns=["season", "round_number", "home_team", "away_team"]))

    final_mask = (
        model_data["season"].eq("2025-26")
        & model_data["home_team"].eq("Paris")
        & model_data["away_team"].eq("Arsenal")
        & model_data["round_number"].eq(17)
    )
    final_match = model_data.loc[final_mask].copy()

    if final_match.empty:
        raise ValueError("Could not find the 2026 final row for Paris vs Arsenal.")

    final_probability = neutral_venue_probability(
        trace=trace,
        row=final_match.iloc[0],
        team_a="Paris",
        team_b="Arsenal",
        mean=mean,
        std=std,
        season_to_idx=season_to_idx,
        team_to_idx=team_to_idx,
    )
    prediction = "Paris" if final_probability >= 0.5 else "Arsenal"

    print("\nPredicted Champions League 2026 final winner:")
    print(prediction)
    print("\nPosterior mean probabilities (neutral venue adjusted):")
    print(f"Paris win probability: {final_probability:.4f}")
    print(f"Arsenal win probability: {1.0 - final_probability:.4f}")


if __name__ == "__main__":
    main()
