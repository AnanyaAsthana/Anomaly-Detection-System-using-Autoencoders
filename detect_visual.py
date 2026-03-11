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

threshold = 0.25
WINDOW_SIZE      = 100
vehicle_buffers  = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
prev_speeds      = {}
flagged_vehicles = set()
bad_driver_id    = None

def get_error(window_data):
    scaled = scaler.transform(window_data)
    X = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        X_pred = model(X)
    return torch.mean((X - X_pred) ** 2).item()

# ── Launch with GUI ────────────────────────────────────────
traci.start(["sumo-gui", "-c", "highway.sumocfg",
             "--delay", "100"])  # slow enough to watch

step = 0
print("Simulation started! Press Play in the SUMO GUI window.")
print(f"Threshold = {threshold}")
print("-" * 60)

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    current_time = round(step * 0.1, 1)
    active = traci.vehicle.getIDList()

    # Color all normal cars YELLOW
    for veh_id in active:
        if veh_id != bad_driver_id and veh_id not in flagged_vehicles:
            traci.vehicle.setColor(veh_id, (255, 255, 0, 255))

    # Pick bad driver at t=50s
    if current_time >= 50 and bad_driver_id is None:
        if len(active) > 2:
            bad_driver_id = active[1]
            traci.vehicle.setColor(bad_driver_id, (255, 0, 0, 255))
            print(f"\n>>> BAD DRIVER: {bad_driver_id} (RED car) <<<\n")

    # Make bad driver erratic
    if bad_driver_id and bad_driver_id in active:
        if int(step) % 10 == 0:
            traci.vehicle.setSpeed(bad_driver_id, 40.0)
        elif int(step) % 10 == 5:
            traci.vehicle.setSpeed(bad_driver_id, 0.0)

    # Collect data
    for veh_id in active:
        speed      = traci.vehicle.getSpeed(veh_id)
        lane_pos   = traci.vehicle.getLanePosition(veh_id)
        lane_idx   = traci.vehicle.getLaneIndex(veh_id)
        prev_speed = prev_speeds.get(veh_id, speed)
        accel      = (speed - prev_speed) / 0.1
        prev_speeds[veh_id] = speed
        vehicle_buffers[veh_id].append([speed, accel, lane_pos, lane_idx])

    # Anomaly detection every 1 second
    if step % 10 == 0 and step > WINDOW_SIZE:
        for veh_id in list(active):
            buf = vehicle_buffers[veh_id]
            if len(buf) == WINDOW_SIZE:
                error = get_error(np.array(buf))

                if veh_id == bad_driver_id:
                    status = "ANOMALY ⚠️" if error > threshold else "Normal ✅"
                    print(f"t={current_time:6.1f}s | BAD_DRIVER | Error: {error:.4f} | {status}")

                if error > threshold and veh_id not in flagged_vehicles:
                    flagged_vehicles.add(veh_id)
                    # Turn flagged cars ORANGE
                    traci.vehicle.setColor(veh_id, (255, 165, 0, 255))
                    print(f"t={current_time:6.1f}s | {veh_id} | Error: {error:.4f} | *** FLAGGED ***")

    step += 1

traci.close()
print("\n" + "=" * 60)
print(f"BAD_DRIVER caught: {'YES ✅' if bad_driver_id in flagged_vehicles else 'NO ❌'}")
print(f"Total flagged: {len(flagged_vehicles)}")