# server.py  –  Flask API
# -----------------------------------------------------
from flask import Flask, render_template, request, redirect, url_for, session
import filtering
import main
import recommendations          # “דף החכמה” הקיים
# -----------------------------------------------------

app = Flask(__name__)
app.secret_key = "your_secret_key_here"


# ---------------------------------------
# 1️⃣  בחירת קבוצה ראשונית
# ---------------------------------------
@app.route("/", methods=["GET", "POST"])
def select_team():
    if request.method == "POST":
        team_name  = request.form.get("team_name")
        max_budget = request.form.get("max_budget")      # נשמר ב-session
        session["max_budget"] = max_budget
        return redirect(url_for("select_criteria", team_name=team_name))

    teams = sorted(main.team_dict.keys())
    return render_template("select_team.html", teams=teams)


# ---------------------------------------
# 2️⃣  טופס הקריטריונים
# ---------------------------------------
@app.route("/criteria/<team_name>", methods=["GET", "POST"])
def select_criteria(team_name):
    if request.method == "POST":
        form_data = request.form.to_dict()

        # משתמש עדכן תקציב?  שומרים ב-session
        if form_data.get("max_budget"):
            try:
                session["max_budget"] = int(form_data["max_budget"])
            except ValueError:
                session["max_budget"] = None

        return redirect(url_for("results", team_name=team_name, **form_data))

    # GET – מכינים נתונים לטופס
    max_budget    = session.get("max_budget")
    positions     = sorted(main.final_df["position"].dropna().unique())
    clubs         = sorted(main.final_df["club name"].dropna().unique())
    nationalities = sorted(main.final_df["country of citizenship"].dropna().unique())

    return render_template(
        "select_criteria.html",
        team_name=team_name,
        positions=positions,
        clubs=clubs,
        max_budget=max_budget,
        nationalities=nationalities
    )


# ---------------------------------------
# 3️⃣  תוצאות – חיפוש שחקנים
# ---------------------------------------
@app.route("/results/<team_name>")
def results(team_name):

    def parse_param(name, cast):
        v = request.args.get(name)
        try:
            return None if v in (None, "") else cast(v)
        except ValueError:
            return None

    position = request.args.get("position")
    if not position:
        return "Position must be selected!", 400

    criteria = {
        "team_name"  : team_name,
        "position"   : position,
        "min_age"    : parse_param("min_age", int),
        "max_age"    : parse_param("max_age", int),
        "max_budget" : parse_param("max_budget", int),
        "min_height" : parse_param("min_height", int),
        "max_height" : parse_param("max_height", int),
        "preferred_foot" : request.args.get("preferred_foot") or None,
        "nationality"    : request.args.get("nationality")    or None,
        "min_market_val" : parse_param("min_market_val", int),
        "max_market_val" : parse_param("max_market_val", int),
        "skill_moves"    : parse_param("skill_moves", int),
        "weak_foot"      : parse_param("weak_foot",  int),
        "min_contract_exp" : parse_param("min_contract_exp", int),
        "max_contract_exp" : parse_param("max_contract_exp", int),
        "curr_club"        : request.args.get("curr_club") or None,
        "min_similarity" : parse_param("min_similarity", float),
    }

    try:
        df = filtering.filter_players_by_criteria(**criteria)
    except Exception as e:
        return f"Error filtering players: {e}", 500

    players = df.to_dict(orient="records")      # תמיד רשימה, גם אם ריקה
    return render_template("results.html",
                           players=players,
                           team_name=team_name,
                           position=position)



# ---------------------------------------
# 4️⃣  “המלצות חכמות”  (קיים אצלך)
# ---------------------------------------
@app.route("/recommendations/<team_name>")
def recommendations_page(team_name):
    max_budget = session.get("max_budget")
    try:
        pos_recs, nat_recs, hot_recs, sel_pos, feat_keys = recommendations.get_recommendations(
            team_name=team_name,
            max_budget=int(max_budget) if max_budget else None
        )
    except ValueError as e:
        return str(e), 404

    return render_template(
        "recommendations.html",
        team_name=team_name,
        max_budget=max_budget,
        selected_position=sel_pos,
        feature_keys=feat_keys,
        position_recommendations=pos_recs.to_dict(orient="records"),
        nationality_recommendations=nat_recs.to_dict(orient="records"),
        hot_players=hot_recs.to_dict(orient="records")
    )


# ---------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
