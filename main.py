import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from torch.utils.data import random_split


##### create the data loader
class DrivingDataset(Dataset):
    def __int__(self, data_patch):
        data = np.load(data_patch)
        self.states = torch.tensor(data['states'], dtype=torch.float32)
        self.actions = torch.tensor(data['actions'], dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


##################################################


#### define the neural network
class DrivingPolicy(nn.Module):
    def __init__(self, input_dim):
        super(DrivingPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4 binary action outputs
            nn.Sigmoid()  # Output in (0, 1) range for binary classification
        )

    def forward(self, x):
        return self.model(x)
###################################################

#### optimization and loading the data into the network
dataset = DrivingDataset("UE5game")
train_size = int(1 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size,val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

model = DrivingPolicy(input_dim=dataset[0][0].shape[0])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()
############################################################


#### training loop
epochs = 20
for epoch in range(epochs) :
    model.train()
    total_loss = 0
    for states, actions in train_loader:
        preds = model(states)
        loss = criterion(preds, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()


#### get the prediction
def predict_action(model, state_tensor, threshold=0.5):
    model.eval()
    with torch.no_grad():
        output = model(state_tensor.unsqueeze(0))  # Add batch dimension
        return (output > threshold).int().squeeze(0).tolist()





