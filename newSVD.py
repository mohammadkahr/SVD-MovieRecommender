import pandas as pd
import numpy as np


def read_data(movies_file, ratings_file):
    try:
        movies = pd.read_csv(movies_file)
        ratings = pd.read_csv(ratings_file)
    except FileNotFoundError as e:
        raise FileNotFoundError("Could not find file: " + str(e))

    return movies, ratings


def merge_movie_data(user_movie_matrix, movies):
    movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')
    user_movie_matrix = user_movie_matrix.merge(movies, left_on='movieId', right_on='movieId', how='left')
    return user_movie_matrix.dropna(subset=['title'])


def preprocess_data(user_movie_matrix):
    user_movie_matrix = user_movie_matrix.pivot_table(index='userId', columns='movieId', values='rating')
    user_movie_matrix = user_movie_matrix.fillna(0)
    user_movie_matrix.columns = user_movie_matrix.columns.astype(int)
    return user_movie_matrix


def svd(matrix, tolerance=1e-10):
    # Compute covariance matrix
    covariance_matrix = np.dot(matrix.T, matrix)

    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Calculate singular values and U, V matrices
    singular_values = np.sqrt(np.abs(eigenvalues))
    U = matrix.dot(eigenvectors) / singular_values
    VT = eigenvectors.T

    # Ensure correct shapes and handle numerical errors
    U = np.nan_to_num(U)
    VT = np.nan_to_num(VT)

    # Apply tolerance to singular values
    mask = singular_values > tolerance
    singular_values = singular_values[mask]
    U = U[:, mask]
    VT = VT[mask, :]

    return U, singular_values, VT


def recommendation_system(user_id, movies, U, Sigma, VT, num_recommendations=10):
    if user_id > U.shape[0]:
        raise ValueError("User ID is out of range")

    user_ratings = np.dot(U[user_id - 1, :] * Sigma, VT)
    recommended_movies_idx = np.argsort(user_ratings)[::-1][:num_recommendations]

    recommendations_df = movies.loc[movies['movieId'].isin(recommended_movies_idx)]
    recommendations_string = recommendations_df[['title', 'genres']].to_string(index=False)

    return recommendations_string


def main():
    movies, ratings = read_data('movies.csv', 'ratings.csv')
    user_movie_matrix = merge_movie_data(ratings, movies)
    user_movie_matrix = preprocess_data(user_movie_matrix)

    if user_movie_matrix.isnull().values.any():
        print("Could not generate recommendations. Please check the input files and try again.")

    else:
        U, Sigma, Vt = svd(user_movie_matrix.values)
        if U is not None and Sigma is not None and Vt is not None:
            try:
                while 5:
                    user_id = int(input("Enter a user ID: "))
                    if user_id == 0: break
                    recommendations = recommendation_system(user_id, movies, U, Sigma, Vt)
                    print(f"Recommended movies for user {user_id}:\n{recommendations}")
            except ValueError as e:
                print(e)
        else:
            print("Could not generate recommendations.")


if __name__ == "__main__":
    main()
