from data_loader import load_movielens_data
from model import GNN
import torch
import torch.optim as optim
import torch.nn.functional as F

# Load data
data = load_movielens_data()

# Initialize the model
model = GNN(in_channels=2, hidden_channels=16, out_channels=2)  # 2 input features, 16 hidden units, 2 output features
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define a simple training loop
def train(data, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        # For now, using random target as placeholder
        target = torch.randn((data.num_nodes, 2))
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Run the training loop
train(data)
