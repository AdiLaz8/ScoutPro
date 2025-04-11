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
print(merged_df.head())
print("Total rows in the DataFrame:", merged_df.shape[0])
print(merged_df.columns.tolist())
