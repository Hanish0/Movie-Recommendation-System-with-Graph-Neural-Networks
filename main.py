from data_loader import load_movielens_data
from model import GNN
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

# Load data and edge labels
data, edge_labels = load_movielens_data()

# Initialize the model
model = GNN(in_channels=20, hidden_channels=16, out_channels=1)  # Single output for regression
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training function
def train(data, edge_labels, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        
        # Compute the loss for edges only
        loss = F.mse_loss(out[data.edge_index[0]], edge_labels)  # MSE for regression
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        # Print loss periodically
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Evaluation function
def evaluate(data, edge_labels):
    model.eval()
    with torch.no_grad():
        predictions = model(data)
        
        # Only evaluate predictions for edges (user-item interactions)
        pred_labels = predictions[data.edge_index[0]].squeeze().cpu().numpy()
        true_labels = edge_labels.cpu().numpy()
        
        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(true_labels, pred_labels)
        print(f"Mean Squared Error (MSE): {mse:.4f}")

# Run the training loop
train(data, edge_labels)

# Evaluate the model
evaluate(data, edge_labels)
