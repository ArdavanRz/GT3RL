import torch
import numpy as np
from Bcloning import PolicyNet
from server import start_server, recv_state, send_action

# Load model
input_dim = 6    # <-- update based on your input size
output_dim = 3   # <-- update for throttle, brake, steering
model = PolicyNet(input_dim, output_dim)
model.load_state_dict(torch.load("bc_policy.pth"))
model.eval()

conn = start_server()

try:
    while True:
        state = recv_state(conn, input_dim)
        state_tensor = torch.from_numpy(state).unsqueeze(0)
        with torch.no_grad():
            action = model(state_tensor).squeeze().numpy()
            action = np.clip(action, -1, 1)
        send_action(conn, action)
except KeyboardInterrupt:
    print("Shutting down.")
    conn.close()