import pandas as pd
import numpy as np


def read_data(movies_file, ratings_file):
    try:
        movies = pd.read_csv(movies_file)
        ratings = pd.read_csv(ratings_file)
    except FileNotFoundError:
        raise FileNotFoundError("Could not find movies.csv or ratings.csv")

    return movies, ratings


def preprocess_data(user_movie_matrix):
    user_movie_matrix = user_movie_matrix.pivot_table(index='userId', columns='movieId', values='rating')
    user_movie_matrix = user_movie_matrix.fillna(0)
    user_movie_matrix.columns = user_movie_matrix.columns.astype(int)
    return user_movie_matrix


def merge_movie_data(user_movie_matrix, movies):
    movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')
    user_movie_matrix = user_movie_matrix.merge(movies, left_on='movieId', right_on='movieId', how='left')
    return user_movie_matrix.dropna(subset=['title'])


def run_svd(matrix, k=610, learning_rate=1, iterations=10):
    m, n = matrix.shape
    U = np.random.normal(scale=1. / k, size=(m, k))
    Sigma = np.diag(np.random.normal(scale=1. / k, size=k))
    V = np.random.normal(scale=1. / k, size=(n, k))

    for i in range(iterations):
        matrix_pred = np.dot(np.dot(U, Sigma), V.T)
        U = U + learning_rate * np.dot((matrix - matrix_pred), V)
        V = V + learning_rate * np.dot((matrix - matrix_pred).T, U)
        Sigma = Sigma + learning_rate * np.diag(np.dot((matrix - matrix_pred).T, U))

    return U, Sigma, V.T


def recommendation_system(user_id, movies, predicted_ratings):
    if user_id > predicted_ratings.shape[0]:
        raise ValueError("User ID is out of range")

    user_ratings = predicted_ratings[user_id - 1, :]
    recommended_movies_idx = np.argsort(user_ratings)[::-1][:10]

    recommendations_df = pd.DataFrame({'movieId': recommended_movies_idx})
    recommendations_df = recommendations_df.merge(movies, on='movieId', how='left')
    recommendations_df = recommendations_df.dropna(subset=['title'])
    recommendations_string = recommendations_df[['title', 'genres']].to_string(index=False)

    return recommendations_string


def main():
    movies, ratings = read_data('movies.csv', 'ratings.csv')
    user_movie_matrix = merge_movie_data(ratings, movies)
    user_movie_matrix = preprocess_data(user_movie_matrix)

    if user_movie_matrix.isnull().values.any():
        print("Could not generate recommendations. Please check the input files and try again.")

    else:
        U, predicted_ratings, Vt = run_svd(user_movie_matrix.values)

        if U is not None and predicted_ratings is not None:
            try:
                user_id = int(input("Enter a user ID: "))
                recommendations = recommendation_system(user_id, movies, predicted_ratings)
                print(f"Recommended movies for user {user_id}:\n{recommendations}")
            except ValueError as e:
                print(e)
        else:
            print("Could not generate recommendations.")


if __name__ == "__main__":
    main()
