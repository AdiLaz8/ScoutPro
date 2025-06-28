# filtering.py
import pandas as pd
from functools import lru_cache
from typing import Optional

import main                    # final_df טעון כאן
import tfidf_processing        # כל פונקציות TF-IDF שלך

# -----------------------------------------------------------
# ①  חישוב TF-IDF  וסימילריות (hybrid)
# -----------------------------------------------------------
@lru_cache(maxsize=1)
def _prepare_similarity():
    """
    מחזיר  token_df  ו-similarity_df היברידי (ממוצע
    בין סיגנאל המועדון הנוכחי לסיגנאל ההעברות).
    """
    token_df = tfidf_processing.prepare_values_for_tokenizing(main.final_df)

    # ==== Current squad ====
    teams_cur = tfidf_processing.tokenize_per_team(token_df)
    players   = tfidf_processing.tokenize_per_player(token_df)
    sim_cur   = tfidf_processing.compute_tfidf_and_similarity(teams_cur, players)

    # ==== Transfers ====
    token_df_transfers = tfidf_processing.prepare_values_for_tokenizing(main.merged_transfers_df)
    teams_tr  = tfidf_processing.tokenize_per_team(token_df_transfers, False)
    sim_tr    = tfidf_processing.compute_tfidf_and_similarity(teams_tr, players)

    # ==== Hybrid  (פשוט ממוצע, אפשר לשנות משקלים) ====
    similarity_df = (sim_cur + sim_tr) / 2.0

    return token_df, similarity_df


def filter_players_by_criteria(
        team_name: str,
        position : str,
        min_contract_exp: Optional[int] = None,
        max_contract_exp: Optional[int] = None,
        curr_club: Optional[int] = None,
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
        weak_foot: Optional[int] = None,
        min_similarity: Optional[float] = None,
) -> pd.DataFrame:

    df = main.final_df.copy()

    # סינון לפי העמדה
    df = df[df["position"] == position]

    # אל תכלול שחקנים שכבר בקבוצה
    df = df[df["club name"].str.lower() != team_name.lower()]

    # --- סינון לפי גיל ---
    if min_age is not None:
        df = df[df["age"] >= min_age]
    if max_age is not None:
        df = df[df["age"] <= max_age]

    # --- סינון לפי תקציב מקסימלי ---
    if max_budget is not None:
        df = df[df["market value in eur"] <= max_budget]

    # --- סינון לפי גובה ---
    if min_height is not None:
        df = df[df["height in cm"] >= min_height]
    if max_height is not None:
        df = df[df["height in cm"] <= max_height]

    # --- רגל מועדפת ---
    if preferred_foot:
        df = df[df["preferred foot"].str.lower() == preferred_foot.lower()]

    # --- לאום ---
    if nationality:
        df = df[df["country of citizenship"] == nationality]

    # --- שווי שוק ---
    if min_market_val is not None:
        df = df[df["market value in eur"] >= min_market_val]
    if max_market_val is not None:
        df = df[df["market value in eur"] <= max_market_val]

    # --- סקילז ---
    if skill_moves is not None:
        df = df[df["skill moves"] >= skill_moves]

    if weak_foot is not None:
        df = df[df["weak foot"] >= weak_foot]

    
    # ודא קודם שהעמודה בפורמט נכון (חד־פעמית)
    df["contract expiration year"] = pd.to_numeric(df["contract expiration year"], errors="coerce")

    # Contract Expiration
    if min_contract_exp is not None:
        df = df[df["contract expiration year"] >= min_contract_exp]
    if max_contract_exp is not None:
        df = df[df["contract expiration year"] <= max_contract_exp]

    # Current Club
    if curr_club:
        df = df[df["club name"].str.strip().str.lower() == curr_club.strip().lower()]
    # אם אחרי הסינון לא נשארו שחקנים – תחזיר טבלה ריקה
    if df.empty:
        return df

    # ==== חישוב סימילריות רק על שחקנים מסוננים ====
    token_df, similarity_df = _prepare_similarity()

    if team_name not in similarity_df.index:
        raise ValueError(f"Team '{team_name}' not found in similarity matrix")

    # מחשב רק לשחקנים שנותרו ב־df
    df["similarity_score"] = similarity_df.loc[team_name][df.index].values

    if min_similarity is not None:
        df = df[df["similarity_score"] >= min_similarity]

    return df.sort_values("similarity_score", ascending=False).reset_index(drop=False)
