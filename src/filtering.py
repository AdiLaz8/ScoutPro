import pandas as pd
import main
from typing import Optional
import urllib.parse

def get_wikipedia_url(name):
    name_parts = name.strip().split()
    url_name = "_".join([part.capitalize() for part in name_parts])
    return f"https://en.wikipedia.org/wiki/{urllib.parse.quote(url_name)}"

def filter_players_by_criteria(
        team_name: str,
        position: str,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        max_budget: Optional[int] = None,
        min_height: Optional[int] = None,
        max_height: Optional[int] = None,
        preferred_foot: Optional[str] = None,
        nationality: Optional[str] = None,
        min_contract_exp: Optional[int] = None,
        max_contract_exp: Optional[int] = None,
        min_market_val: Optional[int] = None,
        max_market_val: Optional[int] = None,
        skill_moves: Optional[int] = None,
        weak_foot: Optional[int] = None,
        curr_club: Optional[str] = None,
        min_final_score: Optional[float] = None
) -> pd.DataFrame:
    if team_name not in main.team_dict:
        raise ValueError(f"Group {team_name} does not exist.")

    if not isinstance(position, str):
        raise ValueError(f"Position {position} must be a string.")

    filtered_df = main.final_df.copy()
    filtered_df = filtered_df[filtered_df['position'] == position]
    filtered_df = filtered_df[filtered_df['club name'] != team_name]
    filtered_df['wikipedia_url'] = filtered_df['name'].apply(get_wikipedia_url)
    filtered_df['name'] = filtered_df['name'].apply(lambda n: " ".join([w.capitalize() for w in str(n).split()]))  # ← אות גדולה


    if min_age is not None:
        filtered_df = filtered_df[filtered_df['age'] >= min_age]
    if max_age is not None:
        filtered_df = filtered_df[filtered_df['age'] <= max_age]

    if any([max_budget is not None, min_market_val is not None, max_market_val is not None]):
        if 'market value in eur' in filtered_df.columns:
            filtered_df['market value in eur'] = pd.to_numeric(filtered_df['market value in eur'], errors='coerce')
        else:
            raise ValueError("Missing 'market value in eur' column in the data!")

    if max_budget is not None:
        filtered_df = filtered_df[filtered_df['market value in eur'] <= max_budget]
    if min_market_val is not None:
        filtered_df = filtered_df[filtered_df['market value in eur'] >= min_market_val]
    if max_market_val is not None:
        filtered_df = filtered_df[filtered_df['market value in eur'] <= max_market_val]

    if min_height is not None:
        filtered_df = filtered_df[filtered_df['height in cm'] >= min_height]
    if max_height is not None:
        filtered_df = filtered_df[filtered_df['height in cm'] <= max_height]

    if preferred_foot is not None:
        preferred_foot = preferred_foot.strip().lower()
        valid_feet = ["right", "left"]
        if preferred_foot not in valid_feet:
            raise ValueError("Preferred foot value must be 'right' or 'left'.")
        filtered_df['preferred foot'] = filtered_df['preferred foot'].astype(str).str.strip().str.lower()
        filtered_df = filtered_df[filtered_df['preferred foot'] == preferred_foot]

    if nationality is not None:
        filtered_df = filtered_df[filtered_df['country of citizenship'] == nationality]

    if 'contract expiration year' in filtered_df.columns:
        filtered_df['contract expiration year'] = pd.to_numeric(
            filtered_df['contract expiration year'], errors='coerce'
        )
        if min_contract_exp is not None:
            filtered_df = filtered_df[filtered_df['contract expiration year'] >= min_contract_exp]
        if max_contract_exp is not None:
            filtered_df = filtered_df[filtered_df['contract expiration year'] <= max_contract_exp]

    if curr_club is not None:
        filtered_df = filtered_df[filtered_df['club name'] == curr_club]

    if skill_moves is not None:
        filtered_df = filtered_df[filtered_df['skill moves'] >= skill_moves]
    if weak_foot is not None:
        filtered_df = filtered_df[filtered_df['weak foot'] >= weak_foot]

    if hasattr(main, "similarity_df"):
        sim_vec = main.similarity_df.loc[team_name]
        filtered_df['similarity_score'] = sim_vec[filtered_df.index].values
        filtered_df = filtered_df.sort_values(by='similarity_score', ascending=False)
    else:
        filtered_df['similarity_score'] = 0

    return filtered_df