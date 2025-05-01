import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

# הגדרה של כל הפיצ'רים הכלליים לכל עמדה לצורך fallback או שימוש כולל
ALL_FEATURES = [
    "acceleration", "sprint speed", "short passing", "long passing", "dribbling",
    "interceptions", "heading accuracy", "standing tackle", "sliding tackle",
    "strength", "stamina", "vision", "long shots", "ball control", "finishing",
    "penalties", "gk reflexes", "gk kicking", "crossing"
]

# תכונות לפי עמדה לחישוב דמיון ו-kmeans
POSITION_FEATURES_KMEANS = {
    "GK": ["gk reflexes", "gk kicking"],
    "CB": ["acceleration", "sprint speed", "short passing", "long passing", "dribbling",
            "interceptions", "heading accuracy", "standing tackle", "sliding tackle",
            "strength", "stamina"],
    "RB": ["acceleration", "sprint speed", "short passing", "long passing", "dribbling",
            "interceptions", "heading accuracy", "standing tackle", "sliding tackle",
            "strength", "stamina", "crossing"],
    "LB": ["acceleration", "sprint speed", "short passing", "long passing", "dribbling",
            "interceptions", "heading accuracy", "standing tackle", "sliding tackle",
            "strength", "stamina", "crossing"],
    "CDM": ["acceleration", "sprint speed", "short passing", "long passing", "dribbling",
             "interceptions", "heading accuracy", "standing tackle", "sliding tackle",
             "strength", "stamina", "vision", "long shots", "ball control"],
    "CM": ["acceleration", "sprint speed", "short passing", "long passing", "dribbling",
            "interceptions", "heading accuracy", "standing tackle", "sliding tackle",
            "strength", "stamina", "vision", "long shots", "ball control"],
    "CAM": ["acceleration", "sprint speed", "short passing", "long passing", "dribbling",
             "stamina", "vision", "long shots", "ball control", "finishing"],
    "LM": ["acceleration", "sprint speed", "short passing", "long passing", "dribbling",
            "stamina", "vision", "long shots", "ball control", "finishing", "crossing"],
    "RM": ["acceleration", "sprint speed", "short passing", "long passing", "dribbling",
            "stamina", "vision", "long shots", "ball control", "finishing", "crossing"],
    "LW": ["acceleration", "sprint speed", "short passing", "long passing", "dribbling",
            "stamina", "vision", "long shots", "ball control", "finishing", "crossing"],
    "RW": ["acceleration", "sprint speed", "short passing", "long passing", "dribbling",
            "stamina", "vision", "long shots", "ball control", "finishing", "crossing"],
    "ST": ["acceleration", "heading accuracy", "sprint speed", "short passing",
            "dribbling", "penalties", "stamina", "vision", "long shots", "ball control", "finishing"]
}

def compute_similarity_to_team(team_df: pd.DataFrame, candidates_df: pd.DataFrame, features: list) -> pd.DataFrame:
    features = [f for f in features if f in team_df.columns and f in candidates_df.columns]
    team_vector = team_df[features].fillna(0).mean().values.reshape(1, -1)
    candidate_matrix = candidates_df[features].fillna(0).values
    similarities = cosine_similarity(candidate_matrix, team_vector).flatten()
    candidates_df = candidates_df.copy()
    candidates_df["similarity_score"] = similarities
    return candidates_df

def run_kmeans_for_position(position: str, df: pd.DataFrame, k: int = 6):
    features = POSITION_FEATURES_KMEANS.get(position)
    if features is None:
        raise ValueError(f"No features defined for position: {position}")

    df_position = df[df["position"] == position].copy()
    missing = [col for col in features if col not in df_position.columns]
    if missing:
        raise ValueError(f"Missing features for KMeans: {missing}")

    df_position[features] = df_position[features].fillna(df_position[features].mean())
    scaler = StandardScaler()
    X = scaler.fit_transform(df_position[features])
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_position["cluster"] = kmeans.fit_predict(X)
    distances = kmeans.transform(X)
    df_position["cluster_center_distance"] = [distances[i][c] for i, c in enumerate(df_position["cluster"])]
    return df_position, kmeans

def get_recommended_cluster_and_candidates(
    position: str,
    team_df: pd.DataFrame,
    full_df_with_clusters: pd.DataFrame,
    kmeans_model,
    features: list,
    top_n: int = 40
) -> pd.DataFrame:
    team_position_df = team_df[team_df["position"] == position].copy()
    team_position_df = team_position_df[features].fillna(team_position_df[features].mean())
    cluster_distances = {}
    for cluster_id in range(kmeans_model.n_clusters):
        center = kmeans_model.cluster_centers_[cluster_id].reshape(1, -1)
        dists = cosine_distances(team_position_df[features], center)
        cluster_distances[cluster_id] = dists.mean()

    sorted_clusters = sorted(cluster_distances.items(), key=lambda x: x[1])
    top_clusters = [cluster_id for cluster_id, _ in sorted_clusters[:3]]

    recommended_candidates = full_df_with_clusters[
        full_df_with_clusters["cluster"].isin(top_clusters)
    ].copy()
    recommended_candidates = recommended_candidates.sort_values(by="cluster_center_distance")
    return recommended_candidates.head(top_n).copy()
