import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity

# Read in the data into a dataframe
train_data = pd.read_csv('./data/data.csv')
# print(train_data.head())

# Clean data and convert Rating column to numeric
train_data['Rating'] = pd.to_numeric(train_data['Rating'], errors='coerce')

# Fill missing values in Rating column with mean ratings
imputer = SimpleImputer(strategy='mean')
train_data['Rating'] = imputer.fit_transform(train_data['Rating'].values.reshape(-1, 1))

# Function to handle irregular data and convert ratings to a standard scale
def preprocess_data(df):
    # Handling irregular data
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    
    # Filling missing values in Rating column with mean ratings
    imputer = SimpleImputer(strategy='mean')
    df['Rating'] = imputer.fit_transform(df['Rating'].values.reshape(-1, 1))
    
    # Normalizing ratings to a scale of 1 to 5
    min_rating = df['Rating'].min()
    max_rating = df['Rating'].max()
    df['Rating'] = 1 + (df['Rating'] - min_rating) * 4 / (max_rating - min_rating)
    
    return df

# Preprocess data
df = preprocess_data(train_data)

# Create a user-movie rating matrix
user_movie_matrix = df.pivot_table(index='User', columns='Movie', values='Rating', aggfunc='mean').fillna(0)

# Calculate similarity matrix using cosine similarity
similarity_matrix = cosine_similarity(user_movie_matrix)

# Function to get movie recommendations for a given user
def get_movie_recommendations(user):
    if user not in user_movie_matrix.index:
        # If user is not in the data, recommend popular movies
        popular_movies = df['Movie'].value_counts().index[:10]
        return pd.DataFrame({'Movie': popular_movies, 'Predicted Rating': [5] * 10})
    
    user_index = user_movie_matrix.index.get_loc(user)
    user_similarities = similarity_matrix[user_index]
    watched_movies = user_movie_matrix.loc[user]
    
    recommendations = pd.DataFrame(columns=['Movie', 'Predicted Rating'])
    
    for movie in user_movie_matrix.columns:
        if watched_movies[movie] == 0:
            movie_ratings = user_movie_matrix[movie]
            predicted_rating = (user_similarities.dot(movie_ratings)) / user_similarities.sum()
            recommendations = recommendations.append({'Movie': movie, 'Predicted Rating': predicted_rating}, ignore_index=True)
    
    recommendations = recommendations.sort_values(by='Predicted Rating', ascending=False)
    
    return recommendations

# Get movie recommendations for a specific user ('Alice' as an example)
recommended_movies = get_movie_recommendations('Alice')
# print(recommended_movies.head(10))

# CLI implementation
def show_recommendations():
    print("Welcome to the Movie Recommendation System!")
    user_name = input("Please enter your name: ")
    
    # Get movie recommendations for the user
    recommendations = get_movie_recommendations(user_name)
    
    # Display recommendations in a user-friendly format
    print("\nHi, {}! Here are your personalized movie recommendations:".format(user_name))
    print("--------------------------------------------------------")
    for index, row in recommendations.iterrows():
        print(f"Movie: {row['Movie']} - Predicted Rating: {row['Predicted Rating']:.2f}")
    print("--------------------------------------------------------")

show_recommendations()