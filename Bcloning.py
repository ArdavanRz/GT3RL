import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Load data
data = np.load("driving_data.npz")
states = data['states'].astype(np.float32)
actions = data['actions'].astype(np.float32)

# Dataset
class DrivingDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.from_numpy(states)
        self.actions = torch.from_numpy(actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

dataset = DrivingDataset(states, actions)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim) 
        )

    def forward(self, x):
        return self.net(x)

input_dim = states.shape[1]
output_dim = actions.shape[1]
model = PolicyNet(input_dim, output_dim)

# Training
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(10):
    epoch_loss = 0
    for state_batch, action_batch in dataloader:
        pred = model(state_batch)
        loss = loss_fn(pred, action_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

# Save model
torch.save(model.state_dict(), "bc_policy.pth")
