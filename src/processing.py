import pandas as pd
from datetime import datetime

def filter_and_process_players(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Filter for players who played in the last season 2024
    df_filtered = df[df['last_season'] == 2024]
    
    # Select specific columns
    columns_of_interest = [
        'player_id',
        'name',
        'last_season',
        'country_of_citizenship',
        'date_of_birth',
        'height_in_cm',
        'contract_expiration_date',
        'current_club_name',
        'market_value_in_eur',
        'highest_market_value_in_eur'
    ]
    
    df_selected = df_filtered[columns_of_interest]
    
    # Convert contract_expiration_date to year only, fill missing with 2027
    df_selected['contract_expiration_date'] = pd.to_datetime(df_selected['contract_expiration_date']).dt.year.fillna(2027).astype(int)
    
    # Calculate age from date_of_birth
    df_selected['date_of_birth'] = pd.to_datetime(df_selected['date_of_birth'])
    df_selected['age'] = (datetime.now() - df_selected['date_of_birth']).dt.days // 365
    
    for column in columns_of_interest:
        df_selected = df_selected.dropna(subset=[column])
    
    return df_selected

# Usage example:
result_df = filter_and_process_players('/Users/adilazarovich/Desktop/Computer_Science/Recoommender_Systems/ScoutPro/data/players.csv')
print("Total rows in the DataFrame:", result_df.shape[0])