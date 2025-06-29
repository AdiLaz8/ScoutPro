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

# Load and process the data.
players_df = processing.filter_and_process_players(get_relative_path('players.csv'))
attributes_df = processing.load_prepare_attributes(get_relative_path('male_players.csv'))
merged_df = processing.merge_players_and_attributes(players_df, attributes_df)
player_stats_df = processing.summarize_player_statistics(get_relative_path('appearances.csv'))
final_df = processing.merge_with_appearances(merged_df, player_stats_df)
merged_transfers_df = processing.process_and_merge_transfers(get_relative_path('transfers.csv'), final_df)
team_dict = processing.create_teams_positions_dict(final_df)

# print("final_df columns:", list(final_df.columns))
# print("merged_transfers_df columns:", list(merged_transfers_df.columns))


print("\n✅ Transfer table successfully merged with player attributes.")
print(f"Number of rows after filtering and merging: {len(merged_transfers_df)}")


print("\n🔍 Sample rows from the merged dataset:")
print(merged_transfers_df[[
    'name', 'to_club_name', 'transfer_season', 'transfer_fee',
    'position' #, 'content_score'
]].head(10).to_string(index=False))

### Now: Compute TF-IDF matrices and cosine similarity scores for teams and players. ###

# First, prepare the final df for tokenizing.

### Case I: Vectorize based on current team ###

print("\nTokenizing Teams and players for applying TF-IDF...")

print("\nCase I: Current teams")

final_df_for_tokenizing = tfidf_processing.prepare_values_for_tokenizing(final_df)
# print("\nColumns in the final for token table:\n", final_df_for_tokenizing.columns.tolist())

print("\nTokenizing teams information, based on current teams...")

teams_info = tfidf_processing.tokenize_per_team(final_df_for_tokenizing)
# print(f"Info for team Futbol Club Barcelona: {teams_info['Futbol Club Barcelona']}")

print("\nTokenizing players information...")
players_info = tfidf_processing.tokenize_per_player(final_df_for_tokenizing)
# some_player = final_df_for_tokenizing.loc[94]
# print(f"Info for player 94: {players_info[94]}")

print("\nTokenizing is done!")

print("\nApplying TF-IDF and computing similarity between teams and players...")
similarity_df = tfidf_processing.compute_tfidf_and_similarity(teams_info, players_info)
# print("Sample from similarity matrix:\n")
# print(similarity_df.head(5).iloc[:, :5])

team_name = 'Juventus Football Club'
position = 'LW'
print(f"\nGetting 5 most similar {position} players for team {team_name}, based on current teams:")
top = tfidf_processing.get_top_k_similar_players(team_name, similarity_df, final_df_for_tokenizing, position)
print(top.to_string(index=False))

### Case II: Vectorize based on teams' lately-transferred players ###

print("\nTokenizing Teams and players for applying TF-IDF...")

print("\nCase II: Transfers")

transfers_df_for_tokenizing = tfidf_processing.prepare_values_for_tokenizing(merged_transfers_df)
# print("\nColumns in the final for token table:\n", transfers_df_for_tokenizing.columns.tolist())

print("\nTokenizing teams information, based on the teams' transfers...")

teams_info_transfers = tfidf_processing.tokenize_per_team(transfers_df_for_tokenizing, False)
# print(f"Info for team Futbol Club Barcelona: {teams_info['Futbol Club Barcelona']}")

print("\nTokenizing is done!")

print("\nApplying TF-IDF and computing similarity between teams and players...")
transfers_similarity_df = tfidf_processing.compute_tfidf_and_similarity(teams_info_transfers, players_info)
# print("Sample from similarity matrix:\n")
# print(similarity_df.head(5).iloc[:, :5])

team_name = 'Juventus Football Club'
position = 'LW'
# print(f"\nGetting 5 most similar {position} players for team {team_name}, based on transfers:")
# top = tfidf_processing.get_top_k_similar_players(team_name, similarity_df, final_df_for_tokenizing, position)
# print(top.to_string(index=False))

### For computing with alpha adjustment: ###
alpha = 0.3
hybrid_similarity_df = tfidf_processing.adjust_alpha_to_similarity(similarity_df, transfers_similarity_df, alpha)
globals().update({
    "final_df": final_df,                       
    "similarity_df": hybrid_similarity_df      
})
# Choose team_name and position
print(f"\nGetting 5 most similar {position} players for team {team_name}, hybrid score:")
top = tfidf_processing.get_top_k_similar_players(team_name, hybrid_similarity_df, final_df_for_tokenizing, position)
print(top.to_string(index=False))

print("\nDone!")