import os
import processing

def get_relative_path(filename):
    """
    הפונקציה מחזירה נתיב יחסי לקובץ בתיקיית הנתונים, יחסית לתיקיית src.
    
    :param filename: שם הקובץ לטעינה.
    :return: נתיב יחסי לקובץ.
    """
    # בונה נתיב יחסי
    relative_path = os.path.join('..', 'data', filename)
    
    return relative_path

players_df = processing.filter_and_process_players(get_relative_path('players.csv'))
attributes_df = processing.load_prepare_attributes(get_relative_path('male_players.csv'))
merged_df = processing.merge_players_and_attributes(players_df, attributes_df)
player_stats_df = processing.summarize_player_statistics(get_relative_path('appearances.csv'))
final_df = processing.merge_with_appearances(merged_df, player_stats_df)
final_df.to_csv("../data/final_players_data.csv", index=False)

print(final_df.head(40))

print("Number of rows and columns in the final DataFrame:", final_df.shape)
print(final_df["goals"].describe())
print(final_df["final_position"].value_counts())
