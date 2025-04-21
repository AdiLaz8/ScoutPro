import pandas as pd
import main
from datetime import datetime
from typing import Optional


def filter_players_by_criteria(
        team_name: str,
        position: str,
        max_age: Optional[int] = None,
        max_budget: Optional[int] = None,
        min_height: Optional[int] = None,
        max_height: Optional[int] = None,
        preferred_foot: Optional[str] = None,
        nationality: Optional[str] = None,
        contract_exp: Optional[int] = None,
        min_market_val: Optional[int] = None,
        max_market_val: Optional[int] = None,
        skill_moves: Optional[int] = None,
        curr_club: Optional[str] = None
        ) -> pd.DataFrame:
    
    if team_name not in main.team_dict:
        raise ValueError(f"Group {team_name} does not exist.")
    
    if not isinstance(position, str):
        raise ValueError(f"Position {position} does not exist.")
    
    filtered_df = main.final_df[main.final_df['position'] == position]

    if max_age is not None:
        if not isinstance(max_age, int):
            raise ValueError("Maximal age value must be an integer.")
        filtered_df = filtered_df[filtered_df['age'] <= max_age]

    if max_budget is not None:
        if not isinstance(max_budget, int):
            raise ValueError("Maximal budget value must be an integer.")
        filtered_df = filtered_df[filtered_df['market_value_in_eur'] <= max_budget]

    if min_height is not None:
        if not isinstance(min_height, int):
            raise ValueError("Minimal height value must be an integer.")
        filtered_df = filtered_df[filtered_df['height_in_cm'] >= min_height]

    if max_height is not None:
        if not isinstance(max_height, int):
            raise ValueError("Maximal height value must be an integer.")
        filtered_df = filtered_df[filtered_df['height_in_cm'] <= max_height]

    if preferred_foot is not None:
        if not (preferred_foot == 'right' or preferred_foot == 'left'):
            raise ValueError("Preferred foot value must be 'right' or 'left'.")
        filtered_df = filtered_df[filtered_df['preferred foot'] == preferred_foot]

    if nationality is not None:
        if not isinstance(nationality, str):
            raise ValueError("Nationality value must be a string.")
        filtered_df = filtered_df[filtered_df['country_of_citizenship'] == nationality]

    if contract_exp is not None:
        if not isinstance(contract_exp, int):
            raise ValueError("Contract expiration year value must be an integer.")
        filtered_df = filtered_df[filtered_df['contract_expiration_date'].dt.year >= contract_exp]

    if min_market_val is not None:
        if not isinstance(min_market_val, int):
            raise ValueError("Minimal market value must be an integer.")
        filtered_df = filtered_df[filtered_df['market_value_in_eur'] >= min_market_val]

    if max_market_val is not None:
        if not isinstance(max_market_val, int):
            raise ValueError("Maximal market value must be an integer.")
        filtered_df = filtered_df[filtered_df['market_value_in_eur'] <= max_market_val]

    if skill_moves is not None:
        if not isinstance(skill_moves, int):
            raise ValueError("Skill moves value must be an integer.")
        if skill_moves > 5 or skill_moves < 1:
            raise ValueError("Skill moves value must be between 1 and 5.")
        filtered_df = filtered_df[filtered_df['skill moves'] == skill_moves]

    if curr_club is not None:
        if curr_club not in main.team_dict:
            raise ValueError(f"Team {curr_club} does not exist.")
        filtered_df = filtered_df[filtered_df['club name'] == curr_club]

    return filtered_df
