import pandas as pd
import main

def get_recommendations(team_name: str):
    if team_name not in main.team_dict:
        raise ValueError(f"Team {team_name} not found.")

    team_positions = main.team_dict[team_name]

    # --- עמדות חסרות ---
    position_counts = {pos: len(players) for pos, players in team_positions.items()}

    needed_position = None
    if position_counts:
        positions_sorted = sorted(position_counts.items(), key=lambda x: x[1])
        if positions_sorted and positions_sorted[0][1] < 4:
            needed_position = positions_sorted[0][0]
        else:
            needed_position = min(position_counts, key=position_counts.get)

    if needed_position:
        candidates_for_position = main.scored_df[
            main.scored_df['position'] == needed_position
        ].copy()

        candidates_for_position['nationality'] = candidates_for_position['country of citizenship']

        candidates_for_position = candidates_for_position.sort_values(
            by='content_score', ascending=False
        ).head(20)
    else:
        candidates_for_position = pd.DataFrame()

    # --- לאום מועדף (נחשב לפי שמות של שחקני הקבוצה) ---
    player_names_in_team = [
        player['name'] for players in team_positions.values() for player in players if 'name' in player
    ]

    if player_names_in_team:
        team_players_df = main.scored_df[main.scored_df['name'].isin(player_names_in_team)]

        if not team_players_df.empty:
            nationality_series = team_players_df['country of citizenship'].dropna()
            if not nationality_series.empty:
                top_nationality = nationality_series.value_counts().idxmax()
            else:
                top_nationality = None
        else:
            top_nationality = None
    else:
        top_nationality = None

    if top_nationality:
        candidates_for_nationality = main.scored_df[
            main.scored_df['country of citizenship'] == top_nationality
        ].copy()

        candidates_for_nationality['nationality'] = candidates_for_nationality['country of citizenship']

        candidates_for_nationality = candidates_for_nationality.sort_values(
            by='content_score', ascending=False
        ).head(20)
    else:
        candidates_for_nationality = pd.DataFrame()

    # --- הכי חמים ---
    if 'bonus_score' in main.scored_df.columns:
        hot_players = main.scored_df.copy()
        hot_players['hotness_score'] = hot_players['content_score'] + main.scored_df['bonus_score']
        hot_players['nationality'] = hot_players['country of citizenship']
        hot_players = hot_players.sort_values(by='hotness_score', ascending=False).head(20)
    else:
        hot_players = main.scored_df.copy()
        hot_players['nationality'] = hot_players['country of citizenship']
        hot_players = hot_players.sort_values(by='content_score', ascending=False).head(20)

    return candidates_for_position, candidates_for_nationality, hot_players