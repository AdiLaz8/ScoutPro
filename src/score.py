import pandas as pd

POSITION_WEIGHTS = {
    "ST": {
        "finishing": 0.40,
        "heading accuracy": 0.10,
        "penalties": 0.05,
        "sprint speed": 0.20,
        "strength": 0.15,
        "acceleration": 0.10
    },
    "CB": {
        "standing tackle": 0.20,
        "sliding tackle": 0.20,
        "interceptions": 0.20,
        "strength": 0.20,
        "sprint speed": 0.20
    },
    "RB": {
        "standing tackle": 0.10,
        "crossing": 0.30,
        "short passing": 0.20,
        "stamina": 0.20,
        "sprint speed": 0.20
    },
    "LB": {
        "standing tackle": 0.10,
        "crossing": 0.30,
        "short passing": 0.20,
        "stamina": 0.20,
        "sprint speed": 0.20
    },
    "CDM": {
        "standing tackle": 0.20,
        "short passing": 0.20,
        "long passing": 0.20,
        "ball control": 0.20,
        "stamina": 0.20
    },
    "CM": {
        "standing tackle": 0.20,
        "short passing": 0.20,
        "long passing": 0.20,
        "ball control": 0.20,
        "stamina": 0.20
    },
    "CAM": {
        "vision": 0.25,
        "short passing": 0.25,
        "long passing": 0.25,
        "long shots": 0.25
    },
    "LM": {
        "dribbling": 0.20,
        "ball control": 0.20,
        "crossing": 0.10,
        "sprint speed": 0.20,
        "finishing": 0.10,
        "stamina": 0.20
    },
    "RM": {
        "dribbling": 0.20,
        "ball control": 0.20,
        "crossing": 0.10,
        "sprint speed": 0.20,
        "finishing": 0.10,
        "stamina": 0.20
    },
    "LW": {
        "dribbling": 0.20,
        "ball control": 0.20,
        "crossing": 0.10,
        "sprint speed": 0.20,
        "finishing": 0.10,
        "stamina": 0.20
    },
    "RW": {
        "dribbling": 0.20,
        "ball control": 0.20,
        "crossing": 0.10,
        "sprint speed": 0.20,
        "finishing": 0.10,
        "stamina": 0.20
    },
    "GK": {
        "gk reflexes": 0.80,
        "gk kicking": 0.20,
    }
}
def compute_content_score(row: pd.Series, position: str) -> float:
    weights = POSITION_WEIGHTS.get(position, {})
    score = 0.0

    for feature, weight in weights.items():
        value = row.get(feature, 0)
        if pd.isna(value):
            value = 0
        score += value * weight

    return score
def compute_bonus(row: pd.Series, position: str) -> float:
    bonus = 0.0
    if "weak foot" in row:
        if row["weak foot"]>3 and position!="GK":
            bonus += (row["weak foot"] / 5) * 0.05
    attacking_positions = ["ST", "CAM", "LW", "RW", "LM", "RM"]
    if position in attacking_positions and "skill moves" in row:
        if row["skill moves"]>3 and position!="GK":
            bonus += (row["skill moves"] / 5) * 0.05

    offensive_positions = ["CAM", "ST", "LW", "RW", "LM", "RM", "CM", "CDM"]
    if position in offensive_positions:
        goals = row.get("goals", 0)
        assists = row.get("assists", 0)
        contribution = goals + assists
        bonus += min(contribution / 20, 1.0) * 0.05
    return bonus
def compute_final_content_score(row: pd.Series, position: str) -> float:
    base_score = compute_content_score(row, position)
    bonus_score = compute_bonus(row, position)
    final_score = base_score + bonus_score
    return (0.9 * base_score) + (0.1 * bonus_score)
def add_scores_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["content_score"] = df.apply(
        lambda row: compute_final_content_score(row, row["position"]),
        axis=1
    )
    return df
def combine_similarity_and_content_score(df: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """
    מחשבת ציון סופי משולב משני מקורות:
    - content_score: ציון מבוסס פיצ'רים לפי position
    - similarity_score: דמיון לשחקני הקבוצה
    
    alpha שולט על המשקל היחסי של כל מרכיב. לדוגמה:
    alpha = 0.7 → 70% content score ו־30% similarity
    """
    df = df.copy()
    df["final_score"] = alpha * df["content_score"] + (1 - alpha) * df["similarity_score"]
    return df




