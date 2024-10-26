import pandas as pd
import torch
from torch_geometric.data import Data

def load_movielens_data():
    # Load the ratings data
    ratings = pd.read_csv("ml-latest-small/ml-latest-small/ratings.csv")
    
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

    # Initialize node features with random values (placeholder)
    num_nodes = len(user_ids) + len(movie_ids)
    node_features = torch.randn((num_nodes, 2))  # Random features (2 per node)

    # Return graph data with node features
    return Data(x=node_features, edge_index=edge_index)
