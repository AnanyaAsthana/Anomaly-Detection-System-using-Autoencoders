import traci
import numpy as np
import pickle
import torch
import torch.nn as nn
from collections import defaultdict, deque
import warnings
warnings.filterwarnings("ignore")

# ── Model Definition ───────────────────────────────────────
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

# ── Load model, scaler, threshold ─────────────────────────
model = LSTMAutoencoder()
model.load_state_dict(torch.load("autoencoder_model.pt"))
model.eval()

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

base_threshold = float(np.load("threshold.npy"))
threshold = 0.25
print(f"Base threshold:     {base_threshold:.6f}")
print(f"Adjusted threshold: {threshold:.6f}")

WINDOW_SIZE      = 100
vehicle_buffers  = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
prev_speeds      = {}
flagged_vehicles = set()
bad_driver_id    = None

def get_reconstruction_error(window_data):
    scaled = scaler.transform(window_data)
    X = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        X_pred = model(X)
    error = torch.mean((X - X_pred) ** 2).item()
    return error

def run_detection():
    global bad_driver_id
    traci.start(["sumo", "-c", "highway.sumocfg"])
    step = 0

    print(f"\nDetection running...")
    print(f"Threshold = {threshold:.6f}")
    print("-" * 65)

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        current_time = round(step * 0.1, 1)
        active_vehicles = traci.vehicle.getIDList()

        # ── Pick a vehicle to be BAD DRIVER at t=50s ──────
        # Wait until t=50 so the chosen vehicle has enough
        # normal history built up first (50s = 500 steps)
        if current_time >= 50 and bad_driver_id is None:
            if len(active_vehicles) > 2:
                bad_driver_id = active_vehicles[1]
                print(f"\n>>> '{bad_driver_id}' chosen as BAD DRIVER at t={current_time}s <<<\n")

        # ── Make BAD DRIVER erratic ────────────────────────
        if bad_driver_id and bad_driver_id in active_vehicles:
            if int(step) % 10 == 0:
                traci.vehicle.setSpeed(bad_driver_id, 40.0)
            elif int(step) % 10 == 5:
                traci.vehicle.setSpeed(bad_driver_id, 0.0)
        elif bad_driver_id and bad_driver_id not in active_vehicles:
            # Vehicle left — pick a new one
            if len(active_vehicles) > 2:
                bad_driver_id = active_vehicles[1]
                print(f"\n>>> New BAD DRIVER: '{bad_driver_id}' at t={current_time}s <<<\n")

        # ── Collect data for all vehicles ──────────────────
        for veh_id in active_vehicles:
            speed      = traci.vehicle.getSpeed(veh_id)
            lane_pos   = traci.vehicle.getLanePosition(veh_id)
            lane_idx   = traci.vehicle.getLaneIndex(veh_id)
            prev_speed = prev_speeds.get(veh_id, speed)
            accel      = (speed - prev_speed) / 0.1
            prev_speeds[veh_id] = speed
            vehicle_buffers[veh_id].append([speed, accel, lane_pos, lane_idx])

        # ── Anomaly check every 1 second ──────────────────
        if step % 10 == 0 and step > WINDOW_SIZE:
            for veh_id in list(active_vehicles):
                buf = vehicle_buffers[veh_id]
                if len(buf) == WINDOW_SIZE:
                    error = get_reconstruction_error(np.array(buf))

                    # Always print bad driver status
                    if veh_id == bad_driver_id:
                        status = "ANOMALY ⚠️ " if error > threshold else "Normal ✅"
                        print(f"t={current_time:6.1f}s | BAD_DRIVER ({veh_id:15s}) | Error: {error:.6f} | {status}")

                    # Flag anomalies first time only
                    if error > threshold and veh_id not in flagged_vehicles:
                        flagged_vehicles.add(veh_id)
                        if veh_id != bad_driver_id:
                            print(f"t={current_time:6.1f}s | {veh_id:20s} | Error: {error:.6f} | *** FLAGGED ***")

        step += 1

    traci.close()
    print("\n" + "=" * 65)
    print("Detection Complete!")
    print(f"Bad driver ID: {bad_driver_id}")
    print(f"Total flagged vehicles: {len(flagged_vehicles)}")

    normal_flagged = [v for v in flagged_vehicles if v != bad_driver_id]
    bad_flagged    = [v for v in flagged_vehicles if v == bad_driver_id]

    print(f"BAD_DRIVER caught:             {'YES ✅' if bad_flagged else 'NO ❌'}")
    print(f"False positives (normal cars): {len(normal_flagged)}")
    if normal_flagged:
        print(f"False positive IDs: {normal_flagged}")

run_detection()