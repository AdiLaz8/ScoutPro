# filtering.py
import pandas as pd
from functools import lru_cache
from typing import Optional

import main                    # final_df טעון כאן
import tfidf_processing        # כל פונקציות TF-IDF שלך
import score                   # compute_final_content_score  וכו'

# -----------------------------------------------------------
# ①  חישוב TF-IDF ↔ Similarity פעם אחת במטמון
# -----------------------------------------------------------
@lru_cache(maxsize=1)
def _prepare_similarity():
    """
    • מכין DataFrame מכווץ (prepare_values_for_tokenizing)
    • מחשב TF-IDF  וקוסיין-סימילריטי
    • מחזיר   token_df , similarity_df
    (מטמון – רץ פעם אחת בכל הרצת השרת)
    """
    token_df = tfidf_processing.prepare_values_for_tokenizing(main.final_df)
    teams_info   = tfidf_processing.tokenize_per_team(token_df)      # לפי current club
    players_info = tfidf_processing.tokenize_per_player(token_df)
    similarity_df = tfidf_processing.compute_tfidf_and_similarity(
        teams_info, players_info
    )
    return token_df, similarity_df


# -----------------------------------------------------------
# ②  פונקציה מרכזית – פילטר + חישוב ציונים
# -----------------------------------------------------------
def filter_players_by_criteria(
        team_name: str,
        position: str,
        # -------- פילטרים סטנדרטיים --------
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        max_budget: Optional[int] = None,
        min_height: Optional[int] = None,
        max_height: Optional[int] = None,
        preferred_foot: Optional[str] = None,
        nationality: Optional[str] = None,
        min_market_val: Optional[int] = None,
        max_market_val: Optional[int] = None,
        skill_moves: Optional[int] = None,
        weak_foot:  Optional[int] = None,
        # -------- פילטרים לפי ציונים --------
        min_similarity: Optional[float] = None,
        min_content_score: Optional[float] = None,
        min_final_score: Optional[float] = None,
        alpha: float = 0.5                     # משקל content לעומת similarity
) -> pd.DataFrame:
    """
    מחזירה DataFrame עם:
        • similarity_score
        • content_score
        • final_score  =  α·content  +  (1-α)·similarity
    ממויין מהגבוה לנמוך ע"פ final_score
    """

    # -------- בסיס הנתונים --------
    df = main.final_df.copy()
    df = df[df['position'] == position]

    # -------- פילטרים "קלאסיים" --------
    if min_age is not None:
        df = df[df['age'] >= min_age]
    if max_age is not None:
        df = df[df['age'] <= max_age]

    if min_height is not None:
        df = df[df['height in cm'] >= min_height]
    if max_height is not None:
        df = df[df['height in cm'] <= max_height]

    if preferred_foot is not None:
        df = df[df['preferred foot'].str.lower() == preferred_foot.lower()]

    if nationality is not None:
        df = df[df['country of citizenship'] == nationality]

    if max_budget is not None:
        df = df[df['market value in eur'] <= max_budget]
    if min_market_val is not None:
        df = df[df['market value in eur'] >= min_market_val]
    if max_market_val is not None:
        df = df[df['market value in eur'] <= max_market_val]

    if skill_moves is not None:
        df = df[df['skill moves'] >= skill_moves]
    if weak_foot is not None:
        df = df[df['weak foot']  >= weak_foot]

    # ---------------------------------------------------
    # חישוב similarity עבור השחקנים שנשארו אחרי הפילטר
    # ---------------------------------------------------
    token_df, similarity_df = _prepare_similarity()

    if team_name not in similarity_df.index:
        raise ValueError(f"Team '{team_name}' not found in similarity matrix")

    # התאמה בין אינדקס df לאינדקס token_df  (זהים כי לא שינינו)
    team_scores = similarity_df.loc[team_name]
    df['similarity_score'] = team_scores[df.index].values

    # ---------------------------------------------------
    # חישוב content + final scores
    # ---------------------------------------------------
    # df['content_score'] = df.apply(
    #     lambda row: score.compute_final_content_score(row, row['position']),
    #     axis=1
    # )
    # df['final_score'] = alpha * df['content_score'] + (1 - alpha) * df['similarity_score']

    # פילטרים אחרונים לפי ציונים
    if min_similarity is not None:
        df = df[df['similarity_score'] >= min_similarity]
    # if min_content_score is not None:
    #     df = df[df['content_score'] >= min_content_score]
    # if min_final_score is not None:
    #     df = df[df['final_score']   >= min_final_score]

    # מיון תוצאה
    df = df.sort_values('similarity_score', ascending=False)

    # נחזיר רק עמודות רלוונטיות (אבל תשאיר מה שצריך לטבלה)
    return df.reset_index(drop=False)   # index  ← player_id
