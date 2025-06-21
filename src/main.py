import os
import pandas as pd
import json
import processing
import tfidf_processing
import score
import kmeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def get_relative_path(filename):
    return os.path.join('..', 'data', filename)


# שלב 1: טעינה ועיבוד נתונים
players_df = processing.filter_and_process_players(get_relative_path('players.csv'))
attributes_df = processing.load_prepare_attributes(get_relative_path('male_players.csv'))
merged_df = processing.merge_players_and_attributes(players_df, attributes_df)
player_stats_df = processing.summarize_player_statistics(get_relative_path('appearances.csv'))
final_df = processing.merge_with_appearances(merged_df, player_stats_df)
merged_transfers_df = processing.process_and_merge_transfers(get_relative_path('transfers.csv'), final_df)

# TODO: We probably don't need the scored_df now
# שלב 2: חישוב ציוני התאמה
# scored_df = score.add_scores_to_dataframe(final_df)
# scored_df["base_score"] = scored_df.apply(lambda row: score.compute_content_score(row, row["position"]), axis=1)
# scored_df["bonus_score"] = scored_df.apply(lambda row: score.compute_bonus(row, row["position"]), axis=1)
# scored_df["rounded_score"] = scored_df["content_score"].round(2)
# merged_transfers_df = processing.process_and_merge_transfers(get_relative_path('transfers.csv'), scored_df)

# # הדפסת טופ שחקנים כלליים
# print("\nשחקנים עם ציוני ההתאמה הכי גבוהים (כולל בסיס ובונוס):")
# print(scored_df[["name", "position", "base_score", "bonus_score", "content_score"]].sort_values(
#     by="content_score", ascending=False).head(20))

# # התפלגות ציונים כללית
# print("\nהתפלגות ציוני ההתאמה (Content Score):")
# score_counts = scored_df["rounded_score"].value_counts().sort_index()
# for score, count in score_counts.items():
#     print(f"ציון {score:.2f}: {count} שחקנים")

# # טופ 10 לפי כל עמדה
# positions = scored_df["position"].unique()
# for pos in sorted(positions):
#     print(f"\n🔹 Top 10 for position: {pos}")
#     top_players = scored_df[scored_df["position"] == pos]
#     top_players = top_players.sort_values(by="content_score", ascending=False).head(10)
#     print(top_players[["name", "content_score"]].to_string(index=False))


# # 🔍 שלב 3: בדיקת המלצות KMeans על קבוצה קיימת
# position = "CB"
# team_name = "Futbol Club Barcelona"  # תוכל לשנות לקבוצה אחרת

# team_dict = processing.create_teams_positions_dict(scored_df)
# if team_name not in team_dict or position not in team_dict[team_name]:
#     print(f"\n❌ אין שחקנים בעמדה {position} בקבוצה {team_name}")
# else:
#     team_position_df = pd.DataFrame(team_dict[team_name][position])

#     # הרצת KMeans על כלל השחקנים בעמדה
#     clustered_df, kmeans_model = kmeans.run_kmeans_for_position(position, scored_df)

#     # קבלת מועמדים מומלצים לפי קלאסטר הקרוב ביותר לסגנון הקבוצתי
#     recommended_df = kmeans.get_recommended_cluster_and_candidates(
#         position=position,
#         team_df=team_position_df,
#         full_df_with_clusters=clustered_df,
#         kmeans_model=kmeans_model,
#         features=kmeans.POSITION_FEATURES_KMEANS[position],
#         top_n=40
#     )

#     # # הדפסת המועמדים המובילים לפי content_score
#     # print(f"\n🧠 Top 20 recommended candidates for {team_name} in position {position}:")
#     # print("עמודות ב־recommended_df:", recommended_df.columns.tolist())

#     # print(recommended_df.sort_values(by="content_score", ascending=False)[
#     #     ["name", "position", "content_score", "cluster", "cluster_center_distance"]
#     # ].head(40).to_string(index=False))

    
# # # גרף PCA של תוצאות ה־KMeans
# # features = kmeans.POSITION_FEATURES_KMEANS[position]
# # X = clustered_df[features]
# # X_scaled = StandardScaler().fit_transform(X)
# # X_pca = PCA(n_components=2).fit_transform(X_scaled)

# # plt.figure(figsize=(10, 6))
# # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clustered_df["cluster"], cmap="tab10", s=30)
# # plt.title(f"PCA Visualization of {position} Clusters")
# # plt.xlabel("PCA Component 1")
# # plt.ylabel("PCA Component 2")
# # plt.colorbar(label="Cluster ID")
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()
# features = kmeans.POSITION_FEATURES_KMEANS[position]

# # 🔍 המשך – שלב חישוב דמיון וציונים משולבים
# features = kmeans.POSITION_FEATURES_KMEANS[position]

# # שלב 1: חישוב ציון דמיון לשחקני הקבוצה (cosine)
# recommended_df = kmeans.compute_similarity_to_team(team_position_df, recommended_df, features)

# # שלב 2: שילוב עם content_score לציון סופי
# recommended_df = score.combine_similarity_and_content_score(recommended_df, alpha=0.3)

# # שלב 3: מיון לפי ציון סופי וסינון לטופ 20
# top_20 = recommended_df.sort_values(by="final_score", ascending=False).head(20)

# # הדפסת התוצאה
# print(f"\n🔝 Top 20 combined recommendations for {team_name} - position {position}:")
# print(top_20[["name", "position", "content_score", "similarity_score", "final_score"]].to_string(index=False))

### Relevant from here ###
print("\n✅ Transfer table successfully merged with player attributes.")
print(f"Number of rows after filtering and merging: {len(merged_transfers_df)}")
# print("Columns in the merged table:", merged_transfers_df.columns.tolist())

# TODO: Delete, added for checking the teams' names and columns for final
# print("Columns in the final table:", final_df.columns.tolist())
# print("Columns in the scored table:", scored_df.columns.tolist())
# print("Teams are: \n", list(team_dict.keys()))

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
similarity_df = tfidf_processing.compute_tfidf_and_similarity(teams_info_transfers, players_info)
# print("Sample from similarity matrix:\n")
# print(similarity_df.head(5).iloc[:, :5])

team_name = 'Juventus'
position = 'LW'
print(f"\nGetting 5 most similar {position} players for team {team_name}, based on transfers:")
top = tfidf_processing.get_top_k_similar_players(team_name, similarity_df, final_df_for_tokenizing, position)
print(top.to_string(index=False))

# TODO: Problematic example: Ajax, RB - Gave a player from AFC Ajax Amsterdam (same team, different names)
# Club names must be uniform!

print("\nDone!")
