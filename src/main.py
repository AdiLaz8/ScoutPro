import os
import pandas as pd
import json
import processing
import tfidf_processing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def get_relative_path(filename):
    return os.path.join('..', 'data', filename)

# Load and process the data
players_df = processing.filter_and_process_players(get_relative_path('players.csv'))
attributes_df = processing.load_prepare_attributes(get_relative_path('male_players.csv'))
merged_df = processing.merge_players_and_attributes(players_df, attributes_df)
player_stats_df = processing.summarize_player_statistics(get_relative_path('appearances.csv'))
final_df = processing.merge_with_appearances(merged_df, player_stats_df)
merged_transfers_df = processing.process_and_merge_transfers(get_relative_path('transfers.csv'), final_df)
team_dict = processing.create_teams_positions_dict(final_df)
final_df_for_tokenizing = tfidf_processing.prepare_values_for_tokenizing(final_df)
teams_info = tfidf_processing.tokenize_per_team(final_df_for_tokenizing)
players_info = tfidf_processing.tokenize_per_player(final_df_for_tokenizing)
similarity_df = tfidf_processing.compute_tfidf_and_similarity(teams_info, players_info)

#Vectorize based on teams' lately-transferred players 
transfers_df_for_tokenizing = tfidf_processing.prepare_values_for_tokenizing(merged_transfers_df)
teams_info_transfers = tfidf_processing.tokenize_per_team(transfers_df_for_tokenizing, False)
transfers_similarity_df = tfidf_processing.compute_tfidf_and_similarity(teams_info_transfers, players_info)

# computing with alpha adjustment
alpha = 0.3
hybrid_similarity_df = tfidf_processing.adjust_alpha_to_similarity(similarity_df, transfers_similarity_df, alpha)
globals().update({
    "final_df": final_df,                       
    "similarity_df": hybrid_similarity_df      
})


