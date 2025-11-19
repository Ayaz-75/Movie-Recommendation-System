import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# -------------------------------
# Load dataset
# -------------------------------
movies = pd.read_csv('data/ml-latest-small/movies.csv')
ratings = pd.read_csv('data/ml-latest-small/ratings.csv')

# Create user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
R = user_item_matrix.to_numpy()

# -------------------------------
# Functions
# -------------------------------

def user_similarity(user1, user2):
    """Compute cosine similarity between two users."""
    return 1 - cosine(user1, user2)

def predict_ratings(target_user_index, R, k=5):
    """
    Predict ratings for target user based on top k similar users
    """
    num_users = R.shape[0]
    similarities = []

    for other_user in range(num_users):
        if other_user != target_user_index:
            sim = user_similarity(R[target_user_index], R[other_user])
            similarities.append((other_user, sim))
    
    # Sort users by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_users = similarities[:k]

    # Predict ratings
    pred_ratings = np.zeros(R.shape[1])
    sim_sum = np.zeros(R.shape[1])

    for user, sim in top_users:
        pred_ratings += sim * R[user]
        sim_sum += sim

    pred_ratings /= sim_sum + 1e-8  # avoid division by zero
    return pred_ratings

def recommend_movies(user_id, R, movies, N=10, k=5):
    """
    Recommend top N movies for the given user_id
    """
    target_index = user_id - 1
    pred_ratings = predict_ratings(target_index, R, k)
    
    # Exclude already rated movies
    rated_movies = R[target_index] > 0
    pred_ratings[rated_movies] = 0

    # Get top N recommendations
    top_indices = np.argsort(pred_ratings)[::-1][:N]
    recommended_movies = movies.loc[movies['movieId'].isin(user_item_matrix.columns[top_indices])]
    scores = pred_ratings[top_indices]

    # Return both titles and predicted scores
    return recommended_movies[['title']], scores

# -------------------------------
# Streamlit App
# -------------------------------

st.title("Movie Recommendation System")
st.write("Get top movie recommendations for any user!")

user_id = st.number_input("Enter User ID", min_value=1, max_value=R.shape[0], value=1)
N = st.number_input("Number of recommendations", min_value=1, max_value=20, value=10)
k = st.number_input("Number of similar users (k)", min_value=1, max_value=50, value=5)

if st.button("Recommend"):
    recommended_movies, scores = recommend_movies(user_id, R, movies, N=N, k=k)
    
    st.subheader(f"Top {N} recommendations for User {user_id}:")
    st.table(recommended_movies)

    # Optional: Visualize predicted ratings
    st.subheader("Predicted Ratings Visualization")
    plt.figure(figsize=(10,6))
    plt.barh(recommended_movies['title'], scores, color='skyblue')
    plt.xlabel("Predicted Rating")
    plt.gca().invert_yaxis()
    st.pyplot(plt)
