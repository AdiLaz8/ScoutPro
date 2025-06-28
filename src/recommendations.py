import pandas as pd
import main
import random

_recommendation_cache = {}

def get_recommendations_tfidf(team_name: str, max_budget: int = None):
    global _recommendation_cache
    cache_key = (team_name, max_budget)

    if team_name not in main.team_dict:
        raise ValueError(f"Team {team_name} not found.")

    team_positions = main.team_dict[team_name]
    all_positions = list(team_positions.keys())

    if not all_positions:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    selected_position = random.choice(all_positions)
    team_players = [p for players in team_positions.values() for p in players]
    team_df = pd.DataFrame(team_players)

    df = main.final_df.copy()
    df = df[df['club name'] != team_name]   # **Exclude players from team**

    # 1. Recommendation by random position
    candidates = df[df['position'] == selected_position].copy()

    if max_budget is not None:
        candidates['market value in eur'] = pd.to_numeric(candidates['market value in eur'], errors='coerce')
        candidates = candidates[candidates['market value in eur'] <= max_budget]

    # similarity_score
    if hasattr(main, "similarity_df"):
        sim_vec = main.similarity_df.loc[team_name]
        candidates['similarity_score'] = sim_vec[candidates.index].values
    else:
        candidates['similarity_score'] = 0

    candidates_for_position = candidates.sort_values(by='similarity_score', ascending=False).head(20)

    # 2. By team's main nationality
    nationality_series = team_df['country of citizenship'].dropna()
    if not nationality_series.empty:
        top_nationality = nationality_series.value_counts().idxmax()
        nat_candidates = df[df['country of citizenship'] == top_nationality].copy()
        if max_budget is not None:
            nat_candidates['market value in eur'] = pd.to_numeric(nat_candidates['market value in eur'], errors='coerce')
            nat_candidates = nat_candidates[nat_candidates['market value in eur'] <= max_budget]
        if hasattr(main, "similarity_df"):
            sim_vec = main.similarity_df.loc[team_name]
            nat_candidates['similarity_score'] = sim_vec[nat_candidates.index].values
        else:
            nat_candidates['similarity_score'] = 0
        candidates_for_nationality = nat_candidates.sort_values(by='similarity_score', ascending=False).head(20)
    else:
        candidates_for_nationality = pd.DataFrame()

    # 3. "Hot players" - all data, top similarity
    hot_players = df.copy()
    if max_budget is not None:
        hot_players['market value in eur'] = pd.to_numeric(hot_players['market value in eur'], errors='coerce')
        hot_players = hot_players[hot_players['market value in eur'] <= max_budget]
    if hasattr(main, "similarity_df"):
        sim_vec = main.similarity_df.loc[team_name]
        hot_players['similarity_score'] = sim_vec[hot_players.index].values
    else:
        hot_players['similarity_score'] = 0
    hot_players = hot_players.sort_values(by='similarity_score', ascending=False).head(20)

    return candidates_for_position, candidates_for_nationality, hot_players, selected_position
