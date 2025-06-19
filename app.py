# Enhanced Movie Recommender with Extra Functionality & Features
import streamlit as st
import pandas as pd
import requests
import re
from functools import lru_cache
from requests.adapters import HTTPAdapter, Retry
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

TMDB_API_KEY = "5e6da2a4939b9b808f90a318802a0ecf"

# Setup TMDB Session with Retry
@lru_cache(maxsize=1024)
def get_tmdb_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

# Clean Movie Title
@lru_cache(maxsize=5000)
def clean_title(title):
    title = re.sub(r"\s\(\d{4}\)$", "", title)
    for prefix in [", The", ", A", ", An"]:
        if prefix in title:
            title = prefix.split(", ")[1] + " " + title.replace(prefix, "")
    return title.strip()

# Get Poster
@lru_cache(maxsize=2048)
def get_poster_url(title):
    if not TMDB_API_KEY:
        return "https://via.placeholder.com/200x300?text=No+Image"
    session = get_tmdb_session()
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    try:
        res = session.get(url, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()
        if data["results"]:
            path = data["results"][0].get("poster_path")
            if path:
                return f"https://image.tmdb.org/t/p/w342{path}"
    except Exception:
        pass
    return "https://via.placeholder.com/200x300?text=No+Image"

# Load Data
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

# Add synthetic user ratings to dataset
def add_user_preferences(ratings_df, movies_df, movie_titles, rating=4.0, user_id=0):
    new_rows = []
    for title in movie_titles:
        movie_id = movies_df[movies_df['title'] == title]['movieId'].values[0]
        new_rows.append({'userId': user_id, 'movieId': movie_id, 'rating': rating})
    return pd.concat([ratings_df, pd.DataFrame(new_rows)], ignore_index=True)

# Train Model
def train_model(ratings):
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    return algo

# Get Recommendations
def get_recommendations(algo, movies, movie_titles, use_posters=True, top_n=12, genre_filter=None):
    movie_ids = movies[movies['title'].isin(movie_titles)]['movieId'].tolist()
    predictions = []
    for mid in movies['movieId'].unique():
        if mid in movie_ids:
            continue
        est = algo.predict(uid=0, iid=mid).est
        predictions.append((mid, est))
    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)

    recs = []
    for mid, est in top_movies:
        row = movies[movies['movieId'] == mid].iloc[0]
        if genre_filter and genre_filter not in row['genres']:
            continue
        clean = clean_title(row['title'])
        poster = get_poster_url(clean) if use_posters else ""
        recs.append({'title': row['title'], 'predicted_rating': est, 'poster_url': poster})
        if len(recs) >= top_n:
            break
    return pd.DataFrame(recs)

# Display Grid Layout
def render_cards(df):
    watch_later = st.session_state.setdefault("watch_later", set())
    liked = st.session_state.setdefault("liked", set())
    disliked = st.session_state.setdefault("disliked", set())

    for i in range(0, len(df), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            if i + j < len(df):
                row = df.iloc[i + j]
                title = row['title']
                with col:
                    st.image(row['poster_url'], width=160)
                    st.markdown(f"**{title}**")
                    st.markdown(f"⭐ {row['predicted_rating']:.2f}")

                    cols2 = st.columns([1, 1, 1])
                    with cols2[0]:
                        if st.button("👍", key=f"like_{i+j}"):
                            liked.add(title)
                            disliked.discard(title)
                    with cols2[1]:
                        if st.button("👎", key=f"dislike_{i+j}"):
                            disliked.add(title)
                            liked.discard(title)
                    with cols2[2]:
                        if st.button("⏰", key=f"watchlater_{i+j}", help="Add to Watch Later"):
                            watch_later.add(title)

    if watch_later:
        st.markdown("### ⏰ Watch Later")
        for movie in watch_later:
            st.markdown(f"- 🎬 {movie}")
    if liked:
        st.markdown("### 👍 Liked Movies")
        for movie in liked:
            st.markdown(f"- 👍 {movie}")
    if disliked:
        st.markdown("### 👎 Disliked Movies")
        for movie in disliked:
            st.markdown(f"- 👎 {movie}")

# CSS and Theme
def style():
    st.markdown("""
        <style>
        .stApp {
            background: radial-gradient(circle at top, #0f2027, #203a43, #2c5364);
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #1e1e1e;
        }
        .css-18e3th9 {
            padding: 3rem 2rem;
        }
        button[kind="primary"] {
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

# Main App
def main():
    style()
    st.title("🍿 Smart Movie Recommender")
    st.markdown("#### Discover movies you'll love, tailored to your taste.")

    movies, ratings = load_data()
    genres = sorted(set(g for genre_list in movies['genres'].str.split('|') for g in genre_list if g != '(no genres listed)'))

    with st.sidebar:
        st.header("🎯 Preferences")
        favorites = st.multiselect("Select your favorite movies:", sorted(movies['title'].tolist()))
        if not favorites:
            st.stop()
        rating = st.slider("Your average rating for them:", 0.5, 5.0, 4.0, 0.5)
        use_posters = st.checkbox("Show Posters", value=True)
        genre_filter = st.selectbox("Filter by Genre (optional):", ["All"] + genres)
        if st.button("🎥 Recommend"):
            st.session_state.go = True
            st.session_state.selected_genre = genre_filter if genre_filter != "All" else None

            ratings_aug = add_user_preferences(ratings, movies, favorites, rating)
            st.session_state.algo = train_model(ratings_aug)

    if st.session_state.get("go") and 'algo' in st.session_state:
        with st.spinner("Fetching recommendations..."):
            recs = get_recommendations(
                st.session_state.algo, movies, favorites, use_posters,
                genre_filter=st.session_state.selected_genre
            )
            st.subheader("🎬 Recommended Movies")
            render_cards(recs)
            st.caption("Data from MovieLens. Poster data via TMDB.")


if __name__ == '__main__':
    main()
