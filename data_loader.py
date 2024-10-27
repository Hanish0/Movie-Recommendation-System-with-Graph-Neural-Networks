import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MultiLabelBinarizer

def load_movielens_data():
    # Load the ratings data and movie metadata
    ratings = pd.read_csv('ml-latest-small/ml-latest-small/ratings.csv')
    movies = pd.read_csv('ml-latest-small/ml-latest-small/movies.csv')

    # Get unique user and movie IDs
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()

    # Map user and movie IDs to node IDs
    user_map = {user_id: i for i, user_id in enumerate(user_ids)}
    movie_map = {movie_id: i + len(user_ids) for i, movie_id in enumerate(movie_ids)}

    # Create edges (user-movie interactions)
    edges = []
    for _, row in ratings.iterrows():
        user_node = user_map[row['userId']]
        movie_node = movie_map[row['movieId']]
        edges.append([user_node, movie_node])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # One-hot encode movie genres
    mlb = MultiLabelBinarizer()
    movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))  # Split genres
    genres_encoded = mlb.fit_transform(movies['genres'])
    
    # Initialize node features for all nodes (users + movies)
    num_users = len(user_ids)
    num_movies = len(movie_ids)
    user_features = torch.randn((num_users, genres_encoded.shape[1]))  # Random features for users
    movie_features = torch.tensor(genres_encoded, dtype=torch.float32)  # One-hot encoded genres for movies

    # Concatenate user and movie features
    node_features = torch.cat([user_features, movie_features], dim=0)

    # Return graph data with node features
    return Data(x=node_features, edge_index=edge_index)
