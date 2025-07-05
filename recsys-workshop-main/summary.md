# Project Summary

## Datasets Used

- **Dataset 1** — kaggle.com [appearances.csv](../data/appearances.csv).
- **Dataset 2** — kaggle.com [male_players.csv](../data/male_players.csv).
- **Dataset 3** — kaggle.com [players.csv](../data/players.csv).
- **Dataset 4** — kaggle.com [transfers.csv](../data/transfers.csv).

Additional data-related information:

- We downloaded all of our datasets from Kaggle, we took **Dataset 3**, which holds all of the players and filtered out retired players, players without clubs and players that have crucial columns missing. Then, we took **Dataset 2**, which has numerical attributes for players taken from FIFA game, and merged it to get numeric attributes on players. Then we used **Dataset 1** which has stats from games the playersd participated in, in order to get a summary of goals and assists for each player in 23/24 season. This computed the final_df.
- In addition, we took all the team that have at least one player and added them to a dictionary that that for each team, computes a list of all of the positions and the players they have in that position.
- We used **Dataset 4** to compute a dataframe that has all of these features and also the team the player moved to, and this computed merged_transfers_df.


&nbsp;<br>

## Technologies and Frameworks

### Frontend

- **HTML + Bootstrap** — Used for creating a responsive and styled user interface, including layout, buttons, and tables across the UI (e.g., landing.html, results.html, recommendations.html).

- **Vanilla JavaScript** — Used for minor client-side interactivity and triggering events, such as auto-scroll or animations (via Animate.css).

### Backend

- **Flask** - Lightweight web framework used to build the backend server, expose RESTful endpoints (e.g., /search, /recommend), and serve HTML templates with dynamic data.


### Algorithmic

- **scikit-learn** — Used for calculating similarity scores between players and teams using TfidfVectorizer and cosine_similarity in content-based recommendation logic, and collaborative filtering.

- **Pandas** — Used extensively for data manipulation, filtering, merging, and analysis across all pipeline stages.



### Data Platforms

- **Git LFS** — Used to store and version large CSV files such as appearances.csv efficiently inside the Git repository.

- **CSV Files** — Raw football data is stored in structured CSV files and loaded using pandas.

&nbsp;<br>

## Main Algorithms

A brief summary of the key algorithms and features developed:

- **Collaborative filtering (Item Based)** — We used the tranfers data to get a recommendation based on recent players the team bought - we computed a vector for the team based on recent transfers.
- **Content-based (TF-IDF)** — We used the final_df dataframe which has for each player the numerical attributes (how they shoot, sprint spleed etc), and computed a vector for each player and vector for each team based on the current players they have on the team, and computed cosine similarity based on TF-IDF matrices.

&nbsp;<br>

## Development Environment
- **VSCode + ChatGPT** - used for the algorithmic modules

&nbsp;<br>

## Development Evolution

Describe the main stages of your system development, major changes, and lessons learned.

Example:

- **Milestone 1:** Collected and consolidated multiple football-related datasets from various public sources online.
- **Milestone 2:**  Invested significant effort in data cleaning, validation, and merging of different CSV files (e.g., attributes, players, appearances, transfers) to create a unified and reliable dataset.
- **Milestone 3:**  Developed the first recommendation engine using KMeans clustering. For each position, we selected five numeric attributes that we believed best represent that role. We weighted them based on perceived importance, computed a custom "score" per player, and added bonuses for qualities such as strong weak foot, high skill moves, many goals or assists, etc. Each team’s cluster was computed based on its existing players, and recommendations were made by finding players closest to that cluster.
- **Milestone 4:** After the mid-project presentation, we integrated transfer market data, and transitioned to a new recommendation engine based on TF-IDF vectorization and cosine similarity, enabling better content-based matching. We also improved the frontend UI to enhance user experience.

&nbsp;<br>

## Open Issues, Limitations, and Future Work
Things we left open/limited due to Miluim of 3 staff members:
- We didn't do a testing to get the right alpha that gets us the most accurate results, so we fixeated it on 80-20 in favor the curren team vector. We did it because the transfers vector was less rich, and with the 80-20 we got results we thought are the most similar to what we know in real life.
- We didn't explain why we chose the recommendations to the user, just gave them the similarity score, due to lack in time.
- Because the domain is focused on the present and a lot of other variables, we can't really calculate how much the recommendations are accurate, only to see the results and analyze them based on us knowing the players and teams.
- Potential next steps:
- Better testing of the alpha, and getting to the optimal alpha
- Adding explainability
- Embedding AI


&nbsp;<br>