import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch.nn.functional import cosine_similarity


data_path = 'C:\\Users\\123\\OneDrive\\桌面\\机器学习大作业\\Project for ML\\MoleculeEvaluationData\\'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load train data
with open(data_path + 'train.pkl\\train.pkl', 'rb') as f:
    train_data = pickle.load(f)

train_fp = np.unpackbits(train_data['packed_fp']).reshape(-1, 2048)
train_cost = train_data['values']
train_fp_tensor = torch.Tensor(train_fp).to(device)
train_cost_tensor = torch.Tensor(train_cost).to(device)

# Load test data
with open(data_path + 'test.pkl\\test.pkl', 'rb') as f:
    test_data = pickle.load(f)

test_fp = np.unpackbits(test_data['packed_fp']).reshape(-1, 2048)
test_cost = test_data['values']
test_fp_tensor = torch.Tensor(test_fp).to(device)


# Define the model architecture
class CostPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super(CostPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define hyperparameters
input_dim = train_fp.shape[1]
hidden_dim_1 = 256
hidden_dim_2 = 64
output_dim = 1
lr = 0.005
num_epochs = 1000

# Initialize the model
model = CostPredictor(input_dim, hidden_dim_1, hidden_dim_2, output_dim).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(torch.Tensor(train_fp_tensor))
    loss = criterion(outputs, torch.Tensor(train_cost_tensor))

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluation
model.eval()
with torch.no_grad():
    test_predictions = model(torch.Tensor(test_fp_tensor))

total_cost = torch.sum(test_predictions).item()
mean_cost = total_cost / len(test_predictions)
test_loss = criterion(test_predictions, torch.Tensor(test_cost).to(device)).item()
print(f'Total predicted cost using Equation 3.1: {total_cost}')
print(f'Mean predicted cost using Equation 3.1: {mean_cost}')
print(f'Test Loss: {test_loss}')

# Define the GNN model
class GNNCostPredictor(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNCostPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.fc1(x))
        x = self.propagate(edge_index, x=x)
        x = self.fc2(x)
        return x

    def message(self, x_i, x_j):
        return x_j


# Define hyperparameters
input_dim = train_fp.shape[1]
hidden_dim = 64
output_dim = 1
lr = 0.005
num_epochs = 100
batch_size = 64

# Prepare data for GNN
train_data = Data(x=torch.Tensor(train_fp), edge_index=torch.tensor([[], []], dtype=torch.long))
test_data = Data(x=torch.Tensor(test_fp), edge_index=torch.tensor([[], []], dtype=torch.long))
train_loader = DataLoader([train_data], batch_size=batch_size, shuffle=True)

# Initialize the GNN model
model = GNNCostPredictor(input_dim, hidden_dim, output_dim).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    for data in train_loader:
        edge_index = data.edge_index.to(device)
        x = data.x.to(device)
        target = torch.Tensor(train_cost).to(device)

        # Forward pass
        out = model(x, edge_index)
        loss = criterion(out, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluation
model.eval()
with torch.no_grad():
    test_data = Data(x=test_fp_tensor, edge_index=torch.tensor([[], []], dtype=torch.long))
    test_data = test_data.to(device)
    test_predictions = model(test_data.x, test_data.edge_index)

total_cost = torch.sum(test_predictions).item()
print(f'Total predicted cost using Equation 3.2: {total_cost}')
