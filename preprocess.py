import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

def preprocess(csv_file="normal_driving_data.csv", window_size=100, step_size=10):
    df = pd.read_csv(csv_file)
    features = ["speed", "acceleration", "lane_position", "lane_index"]
    
    # Normalize
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Save scaler for later
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Create sliding windows per vehicle
    windows = []
    for veh_id, group in df.groupby("vehicle_id"):
        data = group[features].values
        if len(data) < window_size:
            continue
        for start in range(0, len(data) - window_size, step_size):
            window = data[start : start + window_size]
            windows.append(window)
    
    X = np.array(windows)
    print(f"Total windows created: {X.shape[0]}")
    print(f"Each window shape: {X.shape[1:]}")
    print(f"Final dataset shape: {X.shape}")
    np.save("training_data.npy", X)
    print("Saved to training_data.npy!")

preprocess()