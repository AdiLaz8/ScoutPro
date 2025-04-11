import pandas as pd
from datetime import datetime


def normalize_column(series):
    return (series - series.min()) / (series.max() - series.min())

def load_prepare_attributes(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df.columns = [col.strip().lower() for col in df.columns]

    attr = [
        "acceleration", "sprint speed", "finishing", "long shots", "penalties",
        "crossing", "short passing", "long passing", "dribbling", "ball control",
        "interceptions", "heading accuracy", "standing tackle", "sliding tackle",
        "strength", "stamina", "vision", "gk reflexes", "gk kicking",
        "name", "position", "preferred foot", "weak foot", "skill moves","age"
    ]

    general_cols = ["name", "position", "preferred foot", "weak foot", "skill moves","age"]

    df = df.dropna(subset=["name", "position"])
    available_cols = [col for col in attr if col in df.columns]
    df = df[available_cols]

    position_map = {
        "GK": "GK", "CB": "CB", "RB": "RB", "LB": "LB",
        "RWB": "RB", "LWB": "LB",
        "CDM": "CDM", "CM": "CM", "CAM": "CAM",
        "LM": "LM", "RM": "RM",
        "LW": "LW", "RW": "RW", "LF": "LW", "RF": "RW",
        "ST": "ST", "CF": "ST"
    }

    df["final_position"] = df["position"].map(position_map)
    df = df.dropna(subset=["final_position"])

    norm_cols = []
    for col in attr:
        if col in df.columns and col not in general_cols:
            norm_col = col.replace(" ", "_")
            df[norm_col] = normalize_column(df[col])
            norm_cols.append(norm_col)

    return df[["name", "final_position","age", "preferred foot", "weak foot", "skill moves"] + norm_cols]

def filter_and_process_players(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Filter for players who played in the last season 2024
    df_filtered = df[df['last_season'] == 2024]
    
    # Select specific columns
    columns_of_interest = [
        'player_id',
        'name',
        'country_of_citizenship',
        'height_in_cm',
        'contract_expiration_date',
        'current_club_name',
        'market_value_in_eur',
        'highest_market_value_in_eur'
    ]
    
    df_selected = df_filtered[columns_of_interest]
        
    for column in columns_of_interest:
        df_selected = df_selected.dropna(subset=[column])
    df_selected.columns = [col.replace("_", " ") for col in df_selected.columns]
    
    
    return df_selected
def merge_players_and_attributes(players_df: pd.DataFrame, attributes_df: pd.DataFrame) -> pd.DataFrame:

    players_df["name"] = players_df["name"].str.strip().str.lower()
    attributes_df["name"] = attributes_df["name"].str.strip().str.lower()
    players_df = players_df.drop_duplicates(subset=["name"])
    attributes_df = attributes_df.drop_duplicates(subset=["name"])

    
    merged_df = pd.merge(players_df, attributes_df, on="name", how="inner")

    print(f"Merged dataframe contains {merged_df.shape[0]} players with full data.")
    merged_df.columns = [col.replace("_", " ") for col in merged_df.columns]

    return merged_df
# Usage example:
players_df = filter_and_process_players("/Users/jonathan_panov/Downloads/ScoutPro/data/players.csv")
attributes_df = load_prepare_attributes("/Users/jonathan_panov/Downloads/ScoutPro/data/male_players.csv")
merged_df = merge_players_and_attributes(players_df, attributes_df)



print(merged_df.head())

print(merged_df.head())

print("Total rows in the DataFrame:", merged_df.shape[0])
print(merged_df.columns.tolist())

