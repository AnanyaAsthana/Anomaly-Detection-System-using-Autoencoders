import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pickle

# ── Model Definition ──────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, encoding_dim=16):
        super(LSTMAutoencoder, self).__init__()
        # Encoder
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.encoder_fc   = nn.Linear(hidden_size, encoding_dim)
        # Decoder
        self.decoder_fc   = nn.Linear(encoding_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        # Encode
        _, (hidden, _) = self.encoder_lstm(x)
        code = torch.relu(self.encoder_fc(hidden[-1]))
        # Decode
        dec_input = torch.relu(self.decoder_fc(code))
        dec_input = dec_input.unsqueeze(1).repeat(1, x.size(1), 1)
        output, _ = self.decoder_lstm(dec_input)
        return output

# ── Training ──────────────────────────────────────────────
def train():
    X = np.load("training_data.npy").astype(np.float32)

    # Split 80/20
    split = int(0.8 * len(X))
    X_train = torch.tensor(X[:split])
    X_val   = torch.tensor(X[split:])

    train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=32, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val,   X_val),   batch_size=32)

    model     = LSTMAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience, patience_counter = 5, 0
    train_losses, val_losses = [], []

    print("Training started...")
    for epoch in range(50):
        # -- Train --
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # -- Validate --
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += criterion(pred, yb).item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1:02d}/50 | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # -- Early stopping --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "autoencoder_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # -- Plot loss --
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig("training_loss.png")
    print("Loss plot saved to training_loss.png")

    # -- Calculate threshold --
    model.load_state_dict(torch.load("autoencoder_model.pt"))
    model.eval()
    X_train_tensor = torch.tensor(X[:split])
    with torch.no_grad():
        X_pred = model(X_train_tensor)
    errors = torch.mean((X_train_tensor - X_pred) ** 2, dim=(1, 2)).numpy()
    threshold = float(np.mean(errors) + 3 * np.std(errors))
    np.save("threshold.npy", threshold)

    print(f"\nAnomaly Threshold: {threshold:.6f}")
    print("Model saved to autoencoder_model.pt")
    print("Training complete!")

train()