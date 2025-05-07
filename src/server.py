from flask import Flask, render_template, request, redirect, url_for
import filtering
import main
import kmeans
import score
import pandas as pd
import recommendations

app = Flask(__name__)

# עמוד בחירת קבוצה
@app.route("/", methods=["GET", "POST"])
def select_team():
    if request.method == "POST":
        team_name = request.form.get("team_name")
        return redirect(url_for('select_criteria', team_name=team_name))

    teams = list(main.team_dict.keys())
    return render_template("select_team.html", teams=teams)

# עמוד בחירת קריטריונים
@app.route("/criteria/<team_name>", methods=["GET", "POST"])
def select_criteria(team_name):
    if request.method == "POST":
        form_data = request.form.to_dict()
        return redirect(url_for('results', team_name=team_name, **form_data))

    positions = sorted(main.final_df['position'].dropna().unique().tolist())
    clubs = sorted(main.final_df['club name'].dropna().unique().tolist())

    return render_template(
        "select_criteria.html",
        team_name=team_name,
        positions=positions,
        clubs=clubs
    )

# עמוד הצגת תוצאות
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
        "max_age": parse_param("max_age", int),
        "max_budget": parse_param("max_budget", int),
        "min_height": parse_param("min_height", int),
        "max_height": parse_param("max_height", int),
        "preferred_foot": request.args.get("preferred_foot") or None,
        "nationality": request.args.get("nationality") or None,
        "contract_exp": parse_param("contract_exp", int),
        "curr_club": request.args.get("curr_club") or None,
        "skill_moves": parse_param("skill_moves", int)
    }

    # קריאה לאלפא
    alpha = parse_param("alpha", float)
    if alpha is None:
        alpha = 0.3  # ברירת מחדל אם לא סופק

    try:
        filtered_players = filtering.filter_players_by_criteria(**criteria)

        if filtered_players.empty:
            return render_template("results.html", players=[], team_name=team_name, position=position)

        if team_name in main.team_dict and position in main.team_dict[team_name]:
            team_players_list = main.team_dict[team_name][position]
            team_df = pd.DataFrame(team_players_list)
        else:
            team_df = pd.DataFrame()

        features = kmeans.POSITION_FEATURES_KMEANS.get(position, [])

        if not team_df.empty:
            filtered_players = kmeans.compute_similarity_to_team(team_df, filtered_players, features)
        else:
            filtered_players["similarity_score"] = 0

        filtered_players = score.combine_similarity_and_content_score(filtered_players, alpha=alpha)

        players = []
        for _, row in filtered_players.iterrows():
            player = row.to_dict()
            player['final_score'] = row.get('final_score', 0)
            player['content_score'] = row.get('content_score', 0)
            player['similarity_score'] = row.get('similarity_score', 0)
            players.append(player)

    except Exception as e:
        return f"Error filtering players: {str(e)}", 500

    return render_template("results.html", players=players, team_name=team_name, position=position)


@app.route("/recommendations/<team_name>")
def recommendations_page(team_name):
    try:
        pos_recs, nat_recs, hot_recs = recommendations.get_recommendations(team_name)
    except ValueError as e:
        return str(e), 404

    return render_template(
        "recommendations.html",
        team_name=team_name,
        position_recommendations=pos_recs.to_dict(orient="records"),
        nationality_recommendations=nat_recs.to_dict(orient="records"),
        hot_players=hot_recs.to_dict(orient="records")
    )

if __name__ == "__main__":
    app.run(debug=True)