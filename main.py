from data_loader import load_movielens_data
from model import GNN
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data and edge labels
data, train_labels, val_edges, val_labels = load_movielens_data()

# Initialize the model with best hyperparameters
model = GNN(in_channels=20, hidden_channels=64, out_channels=1)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training function
def train(data, edge_labels, model, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        
        # Calculate loss for training edges
        loss = F.mse_loss(out[data.edge_index[0]], edge_labels)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Enhanced evaluation function
def evaluate(data, edge_index, edge_labels, model):
    model.eval()
    with torch.no_grad():
        predictions = model(data)
        
        # Only evaluate predictions for edges
        pred_labels = predictions[edge_index[0]].squeeze().cpu().numpy()
        true_labels = edge_labels.cpu().numpy()
        
        # Calculate evaluation metrics
        mse = mean_squared_error(true_labels, pred_labels)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_labels, pred_labels)
        
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R2): {r2:.4f}")

# Train the model with final configuration
train(data, train_labels, model, optimizer)

# Evaluate the model on the validation set
print("Final Validation Results:")
evaluate(data, val_edges, val_labels, model)
from data_loader import load_movielens_data
from model import GNN
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data and edge labels
data, train_labels, val_edges, val_labels = load_movielens_data()

# Initialize the model with best hyperparameters
model = GNN(in_channels=20, hidden_channels=64, out_channels=1)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training function
def train(data, edge_labels, model, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        
        # Calculate loss for training edges
        loss = F.mse_loss(out[data.edge_index[0]], edge_labels)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Enhanced evaluation function
def evaluate(data, edge_index, edge_labels, model):
    model.eval()
    with torch.no_grad():
        predictions = model(data)
        
        # Only evaluate predictions for edges
        pred_labels = predictions[edge_index[0]].squeeze().cpu().numpy()
        true_labels = edge_labels.cpu().numpy()
        
        # Calculate evaluation metrics
        mse = mean_squared_error(true_labels, pred_labels)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_labels, pred_labels)
        
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R2): {r2:.4f}")

# Train the model with final configuration
train(data, train_labels, model, optimizer)

# Evaluate the model on the validation set
print("Final Validation Results:")
evaluate(data, val_edges, val_labels, model)
