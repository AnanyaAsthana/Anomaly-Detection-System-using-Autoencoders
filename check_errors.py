import traci
import numpy as np
import pickle
import torch
import torch.nn as nn
from collections import defaultdict, deque
import warnings
warnings.filterwarnings("ignore")

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, encoding_dim=16):
        super(LSTMAutoencoder, self).__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.encoder_fc   = nn.Linear(hidden_size, encoding_dim)
        self.decoder_fc   = nn.Linear(encoding_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        code = torch.relu(self.encoder_fc(hidden[-1]))
        dec_input = torch.relu(self.decoder_fc(code))
        dec_input = dec_input.unsqueeze(1).repeat(1, x.size(1), 1)
        output, _ = self.decoder_lstm(dec_input)
        return output

model = LSTMAutoencoder()
model.load_state_dict(torch.load("autoencoder_model.pt"))
model.eval()

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

base_threshold = float(np.load("threshold.npy"))
print(f"Base threshold:       {base_threshold:.6f}")
print(f"x1.5 threshold:       {base_threshold * 1.5:.6f}")
print(f"x1.0 threshold:       {base_threshold * 1.0:.6f}")
print(f"x0.5 threshold:       {base_threshold * 0.5:.6f}")
print("-" * 60)

WINDOW_SIZE     = 100
vehicle_buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
prev_speeds     = {}
bad_errors      = []
normal_errors   = []

def get_error(window_data):
    scaled = scaler.transform(window_data)
    X = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        X_pred = model(X)
    return torch.mean((X - X_pred) ** 2).item()

traci.start(["sumo", "-c", "highway.sumocfg"])
bad_driver_injected = False
step = 0

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    current_time = round(step * 0.1, 1)

    if current_time >= 30 and not bad_driver_injected:
        try:
            traci.vehicle.add("BAD_DRIVER", routeID="route_0", typeID="polite_car")
            bad_driver_injected = True
            print(f"BAD DRIVER injected at t={current_time}s")
        except Exception as e:
            print(f"Injection error: {e}")

    if bad_driver_injected and "BAD_DRIVER" in traci.vehicle.getIDList():
        if int(step) % 10 == 0:
            traci.vehicle.setSpeed("BAD_DRIVER", 40.0)
        elif int(step) % 10 == 5:
            traci.vehicle.setSpeed("BAD_DRIVER", 0.0)

    for veh_id in traci.vehicle.getIDList():
        speed      = traci.vehicle.getSpeed(veh_id)
        lane_pos   = traci.vehicle.getLanePosition(veh_id)
        lane_idx   = traci.vehicle.getLaneIndex(veh_id)
        prev_speed = prev_speeds.get(veh_id, speed)
        accel      = (speed - prev_speed) / 0.1
        prev_speeds[veh_id] = speed
        vehicle_buffers[veh_id].append([speed, accel, lane_pos, lane_idx])

    # Every 1 second collect errors for analysis
    if step % 10 == 0 and step > WINDOW_SIZE:
        for veh_id in list(traci.vehicle.getIDList()):
            buf = vehicle_buffers[veh_id]
            if len(buf) == WINDOW_SIZE:
                error = get_error(np.array(buf))
                if veh_id == "BAD_DRIVER":
                    bad_errors.append(error)
                else:
                    normal_errors.append(error)

    # Stop early after 200 seconds to save time
    if current_time >= 200:
        break

    step += 1

traci.close()

print("\n--- ERROR ANALYSIS ---")
if bad_errors:
    print(f"BAD_DRIVER  errors → Min: {min(bad_errors):.6f} | Max: {max(bad_errors):.6f} | Avg: {np.mean(bad_errors):.6f}")
if normal_errors:
    print(f"Normal cars errors → Min: {min(normal_errors):.6f} | Max: {max(normal_errors):.6f} | Avg: {np.mean(normal_errors):.6f}")

print("\n--- SUGGESTED THRESHOLD ---")
if bad_errors and normal_errors:
    suggested = (np.mean(bad_errors) + np.mean(normal_errors)) / 2
    print(f"Midpoint between avg errors: {suggested:.6f}")
    print(f"Use multiplier: {suggested / base_threshold:.2f}x")