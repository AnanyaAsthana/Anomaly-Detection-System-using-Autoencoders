# 🚗 Real-Time Traffic Anomaly Detection using LSTM Autoencoder + SUMO

A machine learning system that detects anomalous (bad) drivers in real-time traffic simulation using an LSTM Autoencoder trained on normal driving behaviour.

---

## 📌 Project Overview

Traditional traffic monitoring relies on human observation or rule-based systems like speed cameras. These approaches are limited — humans cannot monitor thousands of vehicles simultaneously, and rule-based systems can only catch violations they were explicitly programmed for.

This project solves that by using **unsupervised machine learning**:
- The AI learns what **normal driving** looks like
- During detection, any driver whose behaviour deviates from normal is **automatically flagged**
- No labeled anomaly data is required for training

---

## 🧠 How It Works

```
SUMO Simulation (200 normal cars)
         ↓
TraCI API collects speed, acceleration, lane_position, lane_index
         ↓
Data normalized + split into 10-second sliding windows
         ↓
LSTM Autoencoder trained on normal driving only
         ↓
During detection: reconstruction error calculated per vehicle
         ↓
Error > 0.25 threshold → ANOMALY FLAGGED ⚠️
```

The key insight: the autoencoder learns to reconstruct **normal** driving perfectly. When it encounters **erratic** driving, it cannot reconstruct it accurately — the high reconstruction error triggers the anomaly flag.

---

## 🏗️ Architecture — LSTM Autoencoder

```
Input (batch, 100 timesteps, 4 features)
        ↓
Layer 1: LSTM Encoder       [input=4,  hidden=64]
Layer 2: Dense Encoder      [64 → 16 + ReLU]
Layer 3: Bottleneck         [16 numbers — compressed fingerprint]
Layer 4: Dense Decoder      [16 → 64 + ReLU]
Layer 5: Repeat Vector      [(batch,64) → (batch,100,64)]
Layer 6: LSTM Decoder       [input=64, hidden=4]
        ↓
Output (batch, 100 timesteps, 4 features)
        ↓
MSE Error → < 0.25 Normal ✅  |  > 0.25 Anomaly ⚠️
```

| Layer | Type | Parameters |
|---|---|---|
| LSTM Encoder | LSTM(4→64) | 17,408 |
| Dense Encoder | Linear(64→16) + ReLU | 1,040 |
| Bottleneck | 16 neurons | 0 |
| Dense Decoder | Linear(16→64) + ReLU | 1,088 |
| Repeat Vector | Reshape | 0 |
| LSTM Decoder | LSTM(64→4) | 1,088 |
| **Total** | | **~20,624** |

---

## 📁 File Structure

```
ML_PROJECT/
│
├── 🗺️  SUMO Files
│   ├── highway.net.xml           → Road network (3x3 grid, 3 lanes each)
│   ├── normal_traffic.rou.xml    → Vehicle definitions and routes
│   └── highway.sumocfg           → Master SUMO configuration
│
├── 🐍  Pipeline Scripts
│   ├── collect_data.py           → Step 1: Record driving data from SUMO
│   ├── preprocess.py             → Step 2: Normalize + create sliding windows
│   ├── train_model.py            → Step 3: Build and train LSTM Autoencoder
│   ├── detect_anomaly.py         → Step 4: Real-time detection (terminal only)
│   └── detect_visual.py          → Step 4: Real-time detection (with SUMO GUI)
│
├── 🛠️  Utility Scripts
│   ├── run_visual.py             → Show simulation without AI detection
│   └── check_errors.py           → Diagnostic tool for threshold calibration
│
├── 📊  Data Files
│   ├── normal_driving_data.csv   → Raw recorded driving data (14,000+ rows)
│   ├── training_data.npy         → Processed windows (677, 100, 4)
│   └── training_loss.png         → Training loss graph
│
├── 💾  Saved Model Files
│   ├── autoencoder_model.pt      → Trained AI model weights
│   ├── scaler.pkl                → Saved MinMaxScaler for normalization
│   └── threshold.npy             → Calculated threshold value (0.150438)
│
└── 📄  Documentation
    └── README.md                 → This file
```

---

## ⚙️ Requirements

### System Requirements
- Windows 10 or later
- Python 3.10+ (tested on Python 3.14.2)
- SUMO 1.26.0

### Python Libraries
```
torch>=2.0.0
numpy
pandas
scikit-learn
matplotlib
traci
sumolib
pickle
```

Install all dependencies:
```cmd
pip install torch numpy pandas scikit-learn matplotlib traci sumolib
```

### SUMO Installation
Download SUMO from: https://sumo.dlr.de/docs/Downloads.php

Default install path: `C:\Program Files (x86)\Eclipse\Sumo`

---

## 🚀 How to Run

### Step 0 — Setup
```cmd
cd Desktop\ML_PROJECT
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```

### Step 1 — Collect Normal Driving Data
```cmd
py collect_data.py
```
- Runs SUMO simulation with 200 normal IDM cars
- Records speed, acceleration, lane_position, lane_index every 0.1s
- Output: `normal_driving_data.csv` (~14,000 rows)
- Time: ~1 minute

### Step 2 — Preprocess Data
```cmd
py preprocess.py
```
- Normalizes all features to 0-1 range using MinMaxScaler
- Creates sliding windows of 100 timesteps (10 seconds)
- Output: `training_data.npy` (677 windows), `scaler.pkl`
- Time: ~10 seconds

### Step 3 — Train Model
```cmd
py train_model.py
```
- Builds LSTM Autoencoder (6 layers, ~20,624 parameters)
- Trains for up to 50 epochs with early stopping
- Output: `autoencoder_model.pt`, `threshold.npy`, `training_loss.png`
- Time: ~2-3 minutes

### Step 4 — Run Detection (with visuals)
```cmd
py detect_visual.py
```
- Opens SUMO GUI window — **press Play ▶️**
- At t=50s one car becomes the bad driver (shown in RED)
- AI detects anomaly within ~7 seconds
- Flagged vehicles turn ORANGE

### All Commands at Once
```cmd
cd Desktop\ML_PROJECT
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
py collect_data.py
py preprocess.py
py train_model.py
py detect_visual.py
```

> ⚠️ Run each command and wait for it to finish before running the next one.

---

## 🎨 Visual Color Guide

| Color | Meaning |
|---|---|
| 🟡 Yellow | Normal safe driver |
| 🔴 Red | Bad driver (erratic speed) |
| 🟠 Orange | Flagged as anomaly by AI |

---

## 📊 Features Used

| # | Feature | Source | Why |
|---|---|---|---|
| 1 | Speed | `traci.vehicle.getSpeed()` | Primary indicator of erratic behaviour |
| 2 | Acceleration | Calculated: Δspeed/Δtime | Captures sudden braking and acceleration |
| 3 | Lane Position | `traci.vehicle.getLanePosition()` | Lateral movement within lane |
| 4 | Lane Index | `traci.vehicle.getLaneIndex()` | Which lane the vehicle is in |

---

## 🚗 Vehicle Definitions

### Polite Car (Normal)
| Parameter | Value | Meaning |
|---|---|---|
| accel | 2.6 m/s² | Gentle acceleration |
| decel | 4.5 m/s² | Comfortable braking |
| sigma | 0.1 | Nearly perfect driver |
| maxSpeed | 30 m/s (108 km/h) | Respects speed limit |
| carFollowModel | IDM | Intelligent Driver Model |
| lcCooperative | 1.0 | Fully cooperative lane changes |

### Bad Driver (Anomalous)
| Parameter | Value | Meaning |
|---|---|---|
| accel | 5.0 m/s² | Aggressive acceleration |
| decel | 9.0 m/s² | Emergency level braking |
| sigma | 0.9 | Very erratic behaviour |
| maxSpeed | 50 m/s (180 km/h) | Ignores speed limit |

---

## 📈 Results

| Metric | Value |
|---|---|
| Bad Driver Detected | ✅ YES |
| Detection Time | 7 seconds after anomaly started |
| Reconstruction Error at Detection | 0.266 (threshold: 0.25) |
| Peak Bad Driver Error | 0.347 |
| Normal Car Average Error | 0.107 |
| True Positives | 1 |
| False Positives | 4 |
| False Negatives | 0 |
| Recall | **100%** |
| Precision | 20% |
| F1 Score | 33% |
| Training Loss Reduction | 89% (0.282 → 0.031) |

> The 4 false positives are cars directly behind the bad driver, forced into emergency braking. This is physically explainable and realistic behaviour.

---

## 🔑 Key Concepts

**Why LSTM?**
Driving is time-series data. Patterns only make sense over time — a single hard brake could be normal, but 10 seconds of random speed changes is clearly anomalous. LSTM remembers patterns across the full 10-second window.

**Why Autoencoder?**
No labeled anomaly data is needed. The model only trains on normal driving. Anything it cannot reconstruct accurately is flagged as anomalous.

**Why threshold = 0.25?**
Diagnostic analysis showed normal car errors max out at 0.241 and bad driver errors start at 0.141 with peaks at 0.347. Setting 0.25 catches bad drivers while keeping false positives minimal.

---

## ⚠️ Limitations

- Trained on single road type (3x3 grid)
- Small training dataset (677 windows)
- Threshold manually calibrated
- 4 false positives from physically disrupted cars
- Simulation environment only — not tested on real GPS data

---

## 🔮 Future Improvements

- Add jerk (Δacceleration/Δtime) as a 5th feature
- Train on diverse road types and weather conditions
- Implement adaptive threshold per road environment
- Replace simulation data with real GPS/accelerometer data
- Add anomaly risk score (0-100) instead of binary flag
- Explore Variational Autoencoder (VAE) for better anomaly scoring

---

## 👨‍💻 Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.14.2 | Main programming language |
| PyTorch | 2.10.0 | Deep learning framework |
| SUMO | 1.26.0 | Traffic simulation |
| TraCI | — | Python-SUMO interface |
| NumPy | — | Array operations |
| Pandas | — | Data manipulation |
| Scikit-learn | — | Data preprocessing |
| Matplotlib | — | Visualization |

---

## 📚 References

- SUMO Documentation: https://sumo.dlr.de/docs/
- IDM Paper: Treiber, M., Hennecke, A., & Helbing, D. (2000). Congested traffic states in empirical observations and microscopic simulations.
- LSTM Paper: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
- Autoencoder for Anomaly Detection: Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks.

---

*Built for academic demonstration purposes using SUMO traffic simulation and PyTorch deep learning.*
