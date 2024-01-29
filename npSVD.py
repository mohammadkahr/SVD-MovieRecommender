import pandas as pd
import numpy as np

movies = pd.read_csv('.//movies.csv')
ratings = pd.read_csv('.//ratings.csv')
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
user_movie_matrix = user_movie_matrix.fillna(0)
user_movie_matrix = user_movie_matrix.loc[:, ~user_movie_matrix.columns.astype(str).str.contains('Unnamed')]

U, S, Vt = np.linalg.svd(user_movie_matrix, full_matrices=False)


def recommendation_system(user_id, U, S, Vt, movies, n_recommendations=10):
    predicted_ratings = np.dot(U, np.dot(np.diag(S), Vt))

    user_ratings = predicted_ratings[user_id, :]
    recommended_movies_idx = np.argsort(user_ratings)[::-1][:n_recommendations]
    recommended_movies = movies.loc[movies['movieId'].isin(recommended_movies_idx), 'title']
    recommendations_string = recommended_movies.to_string(index=False)
    return recommendations_string

while 5:
    user_id = int(input("Enter a user ID: "))
    recommendations = recommendation_system(user_id, U, S, Vt, movies)
    print(f"Recommended movies for user {user_id}:\n{recommendations}")