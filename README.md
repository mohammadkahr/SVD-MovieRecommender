# Movie Recommendation System

## Overview

This project is the final assignment for the Applied Linear Algebra course. The aim is to create a movie recommendation system using Singular Value Decomposition (SVD). Recommendation systems are omnipresent in today's world, offering personalized suggestions for movies, music, books, products, and more, thereby enhancing user experience and driving business growth.

## Project Steps

1. **Download Dataset**: Retrieve the Latest MovieLens dataset.
2. **Create User-Movie Rating Matrix**: Construct a matrix where each row represents a user, each column represents a movie, and the elements are the ratings given by users to movies.
3. **SVD Decomposition**: Decompose the user-movie rating matrix using SVD to obtain matrices U, S, and V. Investigate the role of each of these matrices in a recommendation system and implement a recommender using these matrices and cosine similarity. Note that using pre-built SVD implementations is not allowed.
4. **Recommendation Output**: Given a user input, output a list of recommended movies sorted by relevance.

## Implementation

### 1. Using Ready-made SVD Functions
This implementation leverages existing SVD functions from libraries like NumPy or SciPy to perform the decomposition and build the recommendation system.

### 2. Using Gradient Descent
This approach uses gradient descent to approximate the SVD decomposition, allowing for a deeper understanding of the optimization process behind matrix factorization in recommendation systems.

### 3. Normal SVD Implementation
Here, we manually implement the SVD decomposition algorithm, providing insight into the mathematical foundations and computational steps involved in the process.

## Getting Started

### Prerequisites
- Python 3.x
- NumPy
- SciPy (if using ready-made SVD functions)
- Pandas (for handling datasets)

### Conclusion
This project demonstrates the application of SVD in building a movie recommendation system, providing three different implementation approaches to understand and explore various methods in matrix factorization.


## By:
- pouria Talaei 
- Mohammadkazem Harandi

### Contact
For any questions or feedback, please reach out to harandi.mohamma@gmail.com or Talaei.pouria06@gmail.com.
