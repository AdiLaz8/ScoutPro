from flask import Flask, render_template, request, redirect, url_for, session
import filtering
import main
import recommendations
import pandas as pd

app = Flask(__name__)
app.secret_key = "your_secret_key_here"
# ---------  FORMAT MONEY  ---------
def human_value(value):
    """
    950        -> 950
    27 500     -> 28K
    1 200 000  -> 1.2M
    55 000 000 -> 55M
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value                     # אם לא מספר

    if value >= 1_000_000:
        num = value / 1_000_000
        return f"{num:.1f}M".rstrip('0').rstrip('.')  # 1.0M -> 1M
    elif value >= 1_000:
        return f"{round(value / 1_000)}K"
    else:
        return str(value)

# רישום הפילטר בכל ה-templates
app.jinja_env.filters["human_value"] = human_value
# ---------  /FORMAT MONEY  ---------

@app.route("/")
def landing():
    return render_template("landing.html")
@app.route("/select_team", methods=["GET", "POST"])
def select_team():
    if request.method == "POST":
        team_name = request.form.get("team_name")
        session['team_name'] = team_name
        session['selected_filters'] = {}
        action = request.form.get("action")
        if action == "filters" or action is None:
            return redirect(url_for('select_criteria', team_name=team_name))
        elif action == "recs":
            return redirect(url_for('recommendations_page', team_name=team_name))
    teams = list(main.team_dict.keys())
    return render_template("select_team.html", teams=teams)

@app.route("/criteria/<team_name>", methods=["GET", "POST"])
def select_criteria(team_name):
    if request.method == "POST":
        form_data = request.form.to_dict()
        session["selected_filters"] = form_data
        return redirect(url_for('results', team_name=team_name, **form_data))

    selected_filters = session.get("selected_filters", {})
    max_budget = selected_filters.get("max_budget") or session.get("max_budget")
    positions = sorted(main.final_df['position'].dropna().unique().tolist())
    clubs = sorted(main.final_df['club name'].dropna().unique().tolist())
    nationalities = sorted(main.final_df['country of citizenship'].dropna().unique().tolist())

    return render_template(
        "select_criteria.html",
        team_name=team_name,
        positions=positions,
        clubs=clubs,
        max_budget=max_budget,
        nationalities=nationalities,
        selected_filters=selected_filters
    )

@app.route("/results/<team_name>")
def results(team_name):
    position = request.args.get("position")
    if not position:
        return "Position must be selected!", 400

    def parse_param(param_name, cast_func, allow_empty=True):
        value = request.args.get(param_name)
        if value is None or value == "":
            return None if allow_empty else cast_func(0)
        return cast_func(value)

    criteria = {
        "team_name": team_name,
        "position": position,
        "min_age": parse_param("min_age", int),
        "max_age": parse_param("max_age", int),
        "max_budget": parse_param("max_budget", int),
        "min_height": parse_param("min_height", int),
        "max_height": parse_param("max_height", int),
        "preferred_foot": request.args.get("preferred_foot") or None,
        "nationality": request.args.get("nationality") or None,
        "min_contract_exp": parse_param("min_contract_exp", int),
        "max_contract_exp": parse_param("max_contract_exp", int),
        "curr_club": request.args.get("curr_club") or None,
        "skill_moves": parse_param("skill_moves", int),
        "weak_foot": parse_param("weak_foot", int),
        "min_final_score": parse_param("min_final_score", float)
    }
    session["selected_filters"] = {k: v for k, v in criteria.items() if v is not None}

    try:
        filtered_players = filtering.filter_players_by_criteria(**criteria)
        if filtered_players.empty:
            return render_template("results.html", players=[], team_name=team_name, position=position, nationality_counts={})
        players = []
        for _, row in filtered_players.iterrows():
            player = row.to_dict()
            player['similarity_score'] = row.get('similarity_score', 0)
            players.append(player)
        nationality_counts = filtered_players['country of citizenship'].value_counts().to_dict()
    except Exception as e:
        return f"Error filtering players: {str(e)}", 500

    return render_template(
        "results.html", players=players, team_name=team_name, position=position, nationality_counts=nationality_counts
    )

@app.route("/recommendations/<team_name>")
def recommendations_page(team_name):
    max_budget = session.get("max_budget")
    try:
        (pos_recs, nat_recs, hot_recs, selected_position, prospects, expiring, top_nationality,
         nationality_counts) = recommendations.get_recommendations_tfidf(
            team_name=team_name,
            max_budget=int(max_budget) if max_budget else None
        )
    except ValueError as e:
        return str(e), 404

    return render_template(
        "recommendations.html",
        team_name=team_name,
        max_budget=max_budget,
        selected_position=selected_position,
        position_recommendations=pos_recs.to_dict(orient="records"),
        nationality_recommendations=nat_recs.to_dict(orient="records"),
        hot_players=hot_recs.to_dict(orient="records"),
        prospects=prospects.to_dict(orient="records"),
        expiring=expiring.to_dict(orient="records"),
        top_nationality=top_nationality,
        nationality_counts=nationality_counts
    )

if __name__ == "__main__":
    app.run(debug=True)