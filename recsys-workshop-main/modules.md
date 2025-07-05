# Modules Description

## User Interface

- **Technology:** *Flask (Jinja2 templates) + Bootstrap 5 (CDN) + Vanilla JS + Chart.js*  
- **Responsibilities:**  
  - Render every screen:  
    - `landing.html` – lading page background
    - `select_team.html` - main page where we select the team
    - `select_criteria.html` – rich filter form (Bootstrap grid, custom JS pitch-map)  
    - `results.html` – search table with tool-tips & nationality pie-chart  
    - `recommendations.html` – multi-block smart-recommendations dashboard  
  - Collect user inputs and fire POST/GET requests to the backend.  
  - Run **Vanilla JS** for UX tweaks: pitch-button selection, Chart.js pies, tool-tips.  
- **Source code:** [`/landing.html`](../src/templates/landing.html) · [`/select_criteria.html`](../src/templates/select_criteria.html) · [`/results.html`](../src/templates/results.html) · [`/recommendations.html`](../src/templates/recommendations.html) · [`/select_team.html`](../src/templates/select_team.html)

---

## Search Engine

- **Technology:** *pandas (in-memory filters)*  
- **Responsibilities:**  
  - Apply numeric & categorical filters sent from the UI.  
  - Sort the results by the similarity scores already pre-computed in `main.py`.  
- **Interactions:**  
  - Called by `server.py → /results` route → **`filtering.py`** (`filter_players_by_criteria`).  
  - Outputs a filtered `DataFrame`, which `server.py` converts to dictionaries for the Jinja table.  
- **Source code:** [`/filtering.py`](../src/filtering.py) · view logic wired in [`/results.html`](../src/templates/results.html)

---

## Recommendation Engine

- **Technology:** *scikit-learn TF-IDF + cosine similarity*  
- **Responsibilities:**  
  - Build **player & club TF-IDF vectors** (see `tfidf_processing.py`).  
  - For a given club, compute a club-profile vector based on current players on the team and another vector based on recent transfers, and rank players by cosine similarity.  
- **Interactions:**  
  - Exposed at `server.py → /recommendations`.  
  - Uses helper functions in **`recommendations.py`** (wrapping TF-IDF similarity, hot-list, prospects, etc.).  
- **Source code:** [`/tfidf_processing.py`](../src/tfidf_processing.py) · [`/recommendations.py`](../src/recommendations.py)

---

## Item Embedding & Feature Extraction

- **Technology:** *scikit-learn TF-IDF + pandas*  
- **Responsibilities:**  
  - Tokenise free-text (transfer history, club vector, etc.).  
  - Fit a **TF-IDF Vectorizer** and persist the sparse matrices used by the Recommendation Engine.  
- **Source code:** [`/tfidf_processing.py`](../src/tfidf_processing.py)

---

## User Profile Manager

There is **no persistent user-profile component** - the current club & last filters live only in the Flask session (see `server.py`).  
If the server restarts, the session resets.  
- **Source code:** session handling inside [`/server.py`](../src/server.py)

---

## Data Ingestion Pipeline

- **Technology:** *pandas scripts*  
- **Responsibilities:**  
  - Load raw CSVs (`players.csv`, `appearances.csv`, `male_players.csv`, `tranfsers.csv` - large files tracked with **Git LFS**).  
  - Clean, join and produce the master `final_df`.  
- **Source code:** [`/processing.py`](../src/processing.py) [`/main.py`](../src/main.py)

---

## API Gateway / Backend Server

- **Technology:** Flask (Python micro-framework)
- **Responsibilities:**  
Serves as the central orchestrator that connects the UI, search engine, recommendation engine, and data modules.
Exposes HTTP routes that correspond to user actions like team selection, applying filters, and fetching recommendations.
Manages application state (e.g., selected team, filters) using Flask sessions.
Converts backend DataFrames to dictionaries to render via Jinja2 templates in the HTML front-end.
Handles request routing, input validation, and calls to the appropriate data-processing functions.

- **Source code:** [`/server.py`](../src/server.py)