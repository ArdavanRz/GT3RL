import json
import numpy as np

json_path = 'mydata.json'  # your JSON file path

with open(json_path, 'r') as f:
    json_data = json.load(f)

states_list = []
actions_list = []

for key in sorted(json_data.keys(), key=int):  # ensure keys are sorted numerically
    sample = json_data[key]

    # Extract actions
    actions = [sample["throttle"], sample["brake"], sample["steering"]]

    # Convert key (string) to numeric time (int or float)
    time = float(key)

    # Extract states: add time and distance
    state = [time, sample["distance"]]

    states_list.append(state)
    actions_list.append(actions)

# Convert to numpy arrays
states = np.array(states_list, dtype=np.float32)
actions = np.array(actions_list, dtype=np.float32)

# Save processed data
np.savez('driving_data.npz', states=states, actions=actions)
print(f'Saved data with states shape={states.shape} and actions shape={actions.shape}')