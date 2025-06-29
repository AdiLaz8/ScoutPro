import pandas as pd
import main
import random
import urllib.parse

def wikipedia_url(name):
    from urllib.parse import quote
    wiki_name = name.strip().title().replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{quote(wiki_name)}"

def add_wikipedia_links(df):
    df['wikipedia_url'] = df['name'].apply(wikipedia_url)
    return df

def get_recommendations_tfidf(team_name: str, max_budget: int = None):
    if team_name not in main.team_dict:
        raise ValueError(f"Team {team_name} not found.")

    team_positions = main.team_dict[team_name]
    all_positions = list(team_positions.keys())
    if not all_positions:
        return (pd.DataFrame(),) * 8

    # choose random position and random nationality from all df
    df = main.final_df.copy()
    df = add_wikipedia_links(df)
    df = df[df['club name'] != team_name]

    nationality_choices = df['country of citizenship'].dropna().unique().tolist()
    selected_position = random.choice(all_positions)
    top_nationality = random.choice(nationality_choices) if nationality_choices else "Unknown"

    team_players = [p for players in team_positions.values() for p in players]
    team_df = pd.DataFrame(team_players)

    # by position
    candidates = df[df['position'] == selected_position].copy()
    if max_budget is not None:
        candidates['market value in eur'] = pd.to_numeric(candidates['market value in eur'], errors='coerce')
        candidates = candidates[candidates['market value in eur'] <= max_budget]
    if hasattr(main, "similarity_df"):
        sim_vec = main.similarity_df.loc[team_name]
        candidates['similarity_score'] = sim_vec[candidates.index].values
    else:
        candidates['similarity_score'] = 0
    candidates_for_position = candidates.sort_values(by='similarity_score', ascending=False).head(20)
    nationality_counts = candidates_for_position['country of citizenship'].value_counts().to_dict()

    # by random nationality
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

    # hot players
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

    # prospects (u21)
    prospects = df[df['age'] <= 21].copy()
    prospects['prospect'] = True
    if hasattr(main, "similarity_df"):
        sim_vec = main.similarity_df.loc[team_name]
        prospects['similarity_score'] = sim_vec[prospects.index].values
    else:
        prospects['similarity_score'] = 0
    prospects = prospects.sort_values(by='similarity_score', ascending=False).head(20)

    # expiring contracts (<=2026)
    expiring = df[(pd.to_numeric(df['contract expiration year'], errors='coerce') <= 2026)].copy()
    expiring['expiring'] = True
    if hasattr(main, "similarity_df"):
        sim_vec = main.similarity_df.loc[team_name]
        expiring['similarity_score'] = sim_vec[expiring.index].values
    else:
        expiring['similarity_score'] = 0
    expiring = expiring.sort_values(by='similarity_score', ascending=False).head(20)

    return (
        candidates_for_position,
        candidates_for_nationality,
        hot_players,
        selected_position,
        prospects,
        expiring,
        top_nationality,
        nationality_counts
    )