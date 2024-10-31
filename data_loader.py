import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

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

    # Create edges (user-movie interactions) and edge labels (ratings)
    edges = []
    edge_labels = []
    for _, row in ratings.iterrows():
        user_node = user_map[row['userId']]
        movie_node = movie_map[row['movieId']]
        edges.append([user_node, movie_node])
        edge_labels.append(row['rating'])  # Use actual rating as label

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(edge_labels, dtype=torch.float32)

    # Split into training and validation sets
    train_edges, val_edges, train_labels, val_labels = train_test_split(
        edge_index.t().numpy(), edge_labels.numpy(), test_size=0.2, random_state=42
    )
    train_edges, val_edges = torch.tensor(train_edges).t(), torch.tensor(val_edges).t()
    train_labels, val_labels = torch.tensor(train_labels), torch.tensor(val_labels)

    # One-hot encode movie genres
    mlb = MultiLabelBinarizer()
    movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
    genres_encoded = mlb.fit_transform(movies['genres'])

    # Initialize node features
    num_users = len(user_ids)
    num_movies = len(movie_ids)
    user_features = torch.randn((num_users, genres_encoded.shape[1]))
    movie_features = torch.tensor(genres_encoded, dtype=torch.float32)

    # Concatenate user and movie features
    node_features = torch.cat([user_features, movie_features], dim=0)

    # Return graph data with training and validation sets
    return Data(x=node_features, edge_index=train_edges), train_labels, val_edges, val_labels
