import pandas as pd
import main
import random
import kmeans
import score

# זיכרון – שחקנים לפי לאום ושחקנים חמים (נשמר לפי קבוצה ותקציב)
_recommendation_cache = {}

def get_recommendations(team_name: str, max_budget: int = None):
    global _recommendation_cache
    cache_key = (team_name, max_budget)

    if team_name not in main.team_dict:
        raise ValueError(f"Team {team_name} not found.")

    team_positions = main.team_dict[team_name]
    all_positions = list(team_positions.keys())

    if not all_positions:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, []

    selected_position = random.choice(all_positions)
    team_players = [p for players in team_positions.values() for p in players]
    team_df = pd.DataFrame(team_players)

    # ------- המלצה 1: לעמדה רנדומלית (מתעדכן בכל רענון) -------
    candidates = main.scored_df.copy()
    candidates = candidates[candidates['position'] == selected_position]

    if max_budget is not None:
        candidates['market value in eur'] = pd.to_numeric(candidates['market value in eur'], errors='coerce')
        candidates = candidates[candidates['market value in eur'] <= max_budget]

    features = kmeans.POSITION_FEATURES_KMEANS.get(selected_position, [])
    if not team_df.empty:
        candidates = kmeans.compute_similarity_to_team(team_df, candidates, features)
    else:
        candidates['similarity_score'] = 0

    candidates = score.add_component_scores(candidates, selected_position)
    candidates['nationality'] = candidates['country of citizenship'].fillna("N/A")
    candidates_for_position = candidates.sort_values(by='similarity_score', ascending=False).head(20)
    features_for_display = score.get_position_features(selected_position)

    # ------- המלצות 2+3: לפי cache אם קיים -------
    if cache_key in _recommendation_cache:
        cached = _recommendation_cache[cache_key]
        return (
            candidates_for_position,
            cached["nationality"],
            cached["hot"],
            selected_position,
            features_for_display
        )

    # ------- המלצה 2: לפי לאום -------
    nationality_series = team_df['country of citizenship'].dropna()
    if not nationality_series.empty:
        top_nationality = nationality_series.value_counts().idxmax()
        candidates_nat = main.scored_df.copy()
        candidates_nat = candidates_nat[candidates_nat['country of citizenship'] == top_nationality]

        if max_budget is not None:
            candidates_nat['market value in eur'] = pd.to_numeric(candidates_nat['market value in eur'], errors='coerce')
            candidates_nat = candidates_nat[candidates_nat['market value in eur'] <= max_budget]

        if not team_df.empty:
            candidates_nat = kmeans.compute_similarity_to_team(
                team_df,
                candidates_nat,
                kmeans.POSITION_FEATURES_KMEANS.get("CM", [])
            )
        else:
            candidates_nat['similarity_score'] = 0

        candidates_nat = score.add_component_scores(candidates_nat, "CM")
        candidates_nat['component_features'] = candidates_nat['position'].map(score.get_position_features)
        candidates_nat['nationality'] = candidates_nat['country of citizenship'].fillna("N/A")
        candidates_for_nationality = candidates_nat.sort_values(by='similarity_score', ascending=False).head(20)
    else:
        candidates_for_nationality = pd.DataFrame()

    # ------- המלצה 3: שחקנים חמים -------
    hot_players = main.scored_df.copy()

    if max_budget is not None:
        hot_players['market value in eur'] = pd.to_numeric(hot_players['market value in eur'], errors='coerce')
        hot_players = hot_players[hot_players['market value in eur'] <= max_budget]

    if not team_df.empty:
        hot_players = kmeans.compute_similarity_to_team(
            team_df,
            hot_players,
            kmeans.POSITION_FEATURES_KMEANS.get("CM", [])
        )
    else:
        hot_players['similarity_score'] = 0

    hot_players = score.add_component_scores(hot_players, "CM")
    hot_players['component_features'] = hot_players['position'].map(score.get_position_features)
    hot_players['nationality'] = hot_players['country of citizenship'].fillna("N/A")
    hot_players = hot_players.sort_values(by='similarity_score', ascending=False).head(20)

    # ------- שמירה בזיכרון -------
    _recommendation_cache[cache_key] = {
        "nationality": candidates_for_nationality,
        "hot": hot_players
    }

    return candidates_for_position, candidates_for_nationality, hot_players, selected_position, features_for_display
