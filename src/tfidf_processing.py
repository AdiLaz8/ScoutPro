import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def prepare_values_for_tokenizing(final: pd.DataFrame) -> pd.DataFrame:
    """For a given dataframe of players, sanitize and prepare the attributes' names and values for tokenizing."""

    # Use a copy of the dataframe so we won't manipulate the original.
    final_for_token = final.copy()

    # Standardize the columns' names and drop columns you don't need
    final_for_token.columns = [col.replace(' ', '_') for col in final_for_token.columns]
    to_remove = {'transfer_season', 'from_club_name', 'contract_expiration_year', 'goals', 'assists', 'transfer_fee'}
    final_for_token = final_for_token.drop(columns=[col for col in to_remove if col in final_for_token.columns])

    # Separate to numerical attributes, categorical attributes, ordinal attributes and unwanted attributes

    cat_attributes = final_for_token.select_dtypes(include='object').columns.tolist()

    # Things like age, height and capabilities which are rated between 1 and 100
    num_attributes = final_for_token.select_dtypes(include='number').columns.tolist()

    # Capabilities which are rated between 1 to 5
    ord_attributes = {'weak_foot', 'skill_moves'}

    # Exclude ordinal attributes and attributes we don't check, like player's name and his current club
    excluded_attributes = {'to_club_name', 'name', 'club_name', 'position', 'weak_foot', 'skill_moves'}
    cat_attributes = [col for col in cat_attributes if col not in excluded_attributes]
    num_attributes = [col for col in num_attributes if col not in excluded_attributes]
    

    # Numerical attributes which are valid only for goalkeepers
    gk_attributes = {'gk_reflexes', 'gk_kicking'}

    # For every numerical attribute, separate values into 5 bins of value range.
    # We will also standardize the numerical attributes' values
    for col in num_attributes:

        # Goalkeeper attributes hold NaN value whenever player is not a goalkeeper
        if col in gk_attributes:
            
            # Get a boolean vector which represents the locations of non-NaN / NaN values
            non_na_mask = final_for_token[col].notna()
            binned_attribute = pd.qcut(final_for_token.loc[non_na_mask, col], q=5, labels=[f'{col}_{i}' for i in range(1, 6)])
            final_for_token.loc[non_na_mask, f'{col}_binned'] = binned_attribute.astype(str)

        else:
            binned_attribute = pd.qcut(final_for_token[col], q=5, labels=[f'{col}_{i}' for i in range(1, 6)])
            final_for_token[f'{col}_binned'] = binned_attribute.astype(str)

    # Drop the before-binning-numerical-attributes
    final_for_token = final_for_token.drop(columns=num_attributes)

    # Standardize the ordinal attributes' values
    for col in ord_attributes:
        final_for_token[col] = final_for_token[col].astype('Int64').astype(str)

    # Standardize the categorical attributes' values
    for col in cat_attributes:
        final_for_token[col] = final_for_token[col].str.lower().str.replace(' ', '_')

    return final_for_token


def tokenize_per_team(df: pd.DataFrame, per_curr_team: bool = True) -> dict[str, str]:
    """
    For a given dataframe of players, separate to different teams and tokenize attributes.
    If per_curr_team == True, look at the current team. Otherwise, look at the players who were transferred to the team.
    """

    # Get attributes and exclude those we don't check - player's name and his current club
    attributes = df.columns.tolist()
    excluded_attributes = {'to_club_name', 'name', 'club_name', 'position'}
    attributes = [col for col in attributes if col not in excluded_attributes]

    teams_info = {}

    group_by = 'club_name'
    if not per_curr_team:
        group_by = 'to_club_name'

    for team, group in df.groupby(group_by):
        tokens = []
        for attribute in attributes:
            for value in group[attribute].dropna().astype(str):
                token = f'{attribute}={value}'
                tokens.append(token)

        teams_info[team] = ' '.join(tokens)

    return teams_info


def tokenize_per_player(df: pd.DataFrame) -> dict[int, str]:
    """For a given dataframe of players, separate to players and tokenize attributes."""

    # Get attributes and exclude those we don't check - player's name and his current club
    attributes = df.columns.tolist()
    excluded_attributes = {'to_club_name', 'name', 'club_name', 'position'}
    attributes = [col for col in attributes if col not in excluded_attributes]

    players_info = {}
    for _, row in df.iterrows():
        tokens = [f'{attribute}={str(row[attribute])}' for attribute in attributes if pd.notna(row[attribute])]
        players_info[row.name] = ' '.join(tokens)
    
    return players_info


def compute_tfidf_and_similarity(teams_info: dict[str, str], players_info: dict[int, str]) -> pd.DataFrame:
    """
    Compute TF-IDF for every attribute per team and per player, then find similarity.
    The function returns a list of indices for top-k players.
    """

    # Apply TF-IDF
    vectorizer = TfidfVectorizer()
    teams_mat = vectorizer.fit_transform(teams_info.values())
    players_mat = vectorizer.transform(players_info.values())

    # Compute the cosine similarity matrix
    # similarity_mat[i][j] = similarity between team i and player j
    similarity_mat = cosine_similarity(teams_mat, players_mat)
    team_names = list(teams_info.keys())
    player_names = list(players_info.keys())

    # Transform into a dataframe
    similarity_df = pd.DataFrame(similarity_mat, index=team_names, columns=player_names)
    return similarity_df

def adjust_alpha_to_similarity(curr_team_similarity: pd.DataFrame, transfers_similarity: pd.DataFrame, alpha: int = 0.7) -> pd.DataFrame:
    hybrid_similarity_df =  alpha * curr_team_similarity + (1 - alpha) * transfers_similarity
    return hybrid_similarity_df

def get_top_k_similar_players(team_name: str, similarity_df: pd.DataFrame, final_df: pd.DataFrame, position: str, k: int = 5) -> pd.DataFrame:
    """
    Given a team name, a similarity matrix we computed, a position in the team and an integer k,
    find the k players with highest similarity scores for the given team from given position.
    """

    if team_name not in similarity_df.index:
        raise ValueError(f"Error: Team '{team_name}' does not exist.")
    
    # Get scores per player for given team
    team_scores = similarity_df.loc[team_name]
    
    # Ignore players who are already in the given team and players who are not of the given position
    is_not_in_team = final_df['club_name'] != team_name
    is_position = final_df['position'] == position
    mask = is_not_in_team & is_position

    # Get similarity scores of vaild players (who maintain the 'mask' from before)
    valid_players = final_df[mask]
    valid_scores = team_scores[valid_players.index]

    # Get a list of indices which match the 'best' players
    top_k_indices = valid_scores.nlargest(k).index

    # Get the best players and their scores
    top_k = final_df.loc[top_k_indices].copy()
    top_k['similarity_score'] = valid_scores.loc[top_k_indices].values

    return top_k[['name', 'club_name', 'similarity_score']].sort_values(by='similarity_score', ascending=False)