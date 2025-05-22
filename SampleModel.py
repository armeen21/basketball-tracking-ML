import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib

# import ijson for streaming; fall back if unavailable
try:
    import ijson
    USE_IJSON = True
    print("Using ijson for streaming.")
except ImportError:
    USE_IJSON = False
    print("ijson not available; falling back to json.loads streaming.")

# ───── Configuration ──────────────────────────────────────────────────────────
SEED            = 42
FILE_PATH       = 'sampletrackingdata.json'
MAX_FRAMES      = None
SEQ_LEN         = 30  # Reduced from 60 to capture more immediate patterns
BATCH_SIZE      = 64  # Increased from 32 for better gradient estimates
HIDDEN_SIZE     = 256  # Increased from 128 for more model capacity
NUM_LAYERS      = 3    # Increased from 2 for more temporal dependencies
LR              = 5e-4  # Reduced from 1e-3 for more stable training
WEIGHT_DECAY    = 1e-4  # Increased from 1e-5 for better regularization (may help with overfitting)
NUM_EPOCHS      = 30   # Decreased from 50 for faster training time (I would usually use 30-50 to avoid overfitting but this depends on the performance of the model)
VAL_RATIO       = 0.15  # Increased from 0.1 for better validation
TEST_RATIO      = 0.15  # Increased from 0.1 for better testing
GRAD_CLIP       = 0.5   # Reduced from 1.0 for more stable training
INJURY_THRESH   = 50.0  # Threshold for anomaly detection

# ───── Reproducibility ────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def _extract_frame(frame, out_list):
    """
    Extract player and ball positions from a frame.
    Handles both player joint data and ball position data.
    """
    # Unify samples key naming
    samples = frame.get('samples', frame.get('sample', {}))
    players = samples.get('players', [])
    
    if players:
        # Extract player joint positions
        player_joints = []
        for player in players:
            joints = player.get('joints', {})
            # Get coordinates for each joint in sorted order
            joint_coords = [joints[k] for k in sorted(joints.keys())]
            player_joints.append(joint_coords)
        
        # Also get ball position if available
        ball_data = samples.get('ball', [])
        if ball_data and 'pos' in ball_data[0]:
            ball_pos = ball_data[0]['pos']
            # Add ball as an additional "joint"
            player_joints.append([ball_pos])
        
        out_list.append(np.array(player_joints, dtype=np.float32))
    else:
        # Fallback to just ball data if no players
        ball_data = samples.get('ball', [])
        if ball_data and 'pos' in ball_data[0]:
            ball_pos = ball_data[0]['pos']
            # Store ball position as a single "player" with one "joint"
            out_list.append(np.array([[ball_pos]], dtype=np.float32))

def stream_and_parse(file_path, max_frames=None):
    """
    Stream JSONL and parse both player and ball positions.
    """
    print("Reading data file...")
    frames = []
    if USE_IJSON:
        with open(file_path, 'rb') as f:
            parser = ijson.items(f, '', multiple_values=True)
            for idx, frame in enumerate(parser):
                if max_frames and idx >= max_frames:
                    break
                if idx % 1000 == 0:
                    print(f"Processing frame {idx}...")
                _extract_frame(frame, frames)
    else:
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                if max_frames and idx >= max_frames:
                    break
                if idx % 1000 == 0:
                    print(f"Processing frame {idx}...")
                try:
                    frame = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _extract_frame(frame, frames)

    if not frames:
        raise ValueError("No valid tracking data found")
    
    print("Processing frames...")
    data = np.stack(frames)  # shape: (N, P, J, 3) where P=players, J=joints
    N, P, J, C = data.shape
    print(f"Processed {N} frames with {P} players/ball, {J} joints, {C} coordinates")
    
    return data

class TrackingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.input_size = input_size  # P * J * 3 coordinates
        self.hidden_size = hidden_size
        
        # Position encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Main LSTM
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Position decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, self.input_size)
        )
        
        # Anomaly detector
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Flatten input if necessary (B, T, P, J, 3) -> (B, T, P*J*3)
        if x.dim() > 3:
            B, T = x.shape[:2]
            x = x.view(B, T, -1)
        
        # Encode sequence
        x = self.encoder(x)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]  # Take last hidden state
        
        # Generate predictions
        pos_pred = self.decoder(h)
        anomaly_pred = self.classifier(h)
        
        return pos_pred, anomaly_pred

class TrackingDataset(Dataset):
    def __init__(self, data, labels, seq_len):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]
        target = self.data[idx + self.seq_len]
        label = self.labels[idx + self.seq_len]
        
        return seq, target, label

def calculate_metrics(pred, target, anomaly_pred, anomaly_target):
    """Calculate comprehensive accuracy metrics."""
    # Position metrics
    mae = torch.mean(torch.abs(pred - target))
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    
    # Per-dimension RMSE
    rmse_x = torch.sqrt(torch.mean((pred[:, 0::3] - target[:, 0::3]) ** 2))
    rmse_y = torch.sqrt(torch.mean((pred[:, 1::3] - target[:, 1::3]) ** 2))
    rmse_z = torch.sqrt(torch.mean((pred[:, 2::3] - target[:, 2::3]) ** 2))
    
    # Anomaly detection metrics
    anomaly_pred_bool = (anomaly_pred > 0.5).float()
    true_positives = torch.sum((anomaly_pred_bool == 1) & (anomaly_target == 1))
    false_positives = torch.sum((anomaly_pred_bool == 1) & (anomaly_target == 0))
    false_negatives = torch.sum((anomaly_pred_bool == 0) & (anomaly_target == 1))
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'mae': mae.item(),
        'rmse': rmse.item(),
        'rmse_x': rmse_x.item(),
        'rmse_y': rmse_y.item(),
        'rmse_z': rmse_z.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }

def main():
    print("Parsing tracking data...")
    data_4d = stream_and_parse(FILE_PATH, MAX_FRAMES)
    N, P, J, C = data_4d.shape
    feat_dim = P * J * C
    
    # Create labels (based on acceleration)
    # Calculate acceleration across all players/joints
    accel = data_4d[2:] - 2 * data_4d[1:-1] + data_4d[:-2]  # (N-2, P, J, 3)
    mags = np.linalg.norm(accel, axis=3)  # (N-2, P, J)
    max_mags = mags.max(axis=(1,2))  # (N-2,)
    labels = (max_mags > INJURY_THRESH).astype(np.float32)
    labels = np.concatenate(([0, 0], labels))  # (N,)
    
    # Flatten positions for scaling
    flat = data_4d.reshape(N, feat_dim)
    
    # Split indices
    num_windows = N - SEQ_LEN
    n_test = int(num_windows * TEST_RATIO)
    n_val = int(num_windows * VAL_RATIO)
    n_train = num_windows - n_val - n_test
    
    # Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(flat)
    data_scaled = data_scaled.reshape(N, P, J, C)
    
    # Create datasets
    train_dataset = TrackingDataset(
        data_scaled[:n_train + SEQ_LEN],
        labels[:n_train + SEQ_LEN],
        SEQ_LEN
    )
    
    val_dataset = TrackingDataset(
        data_scaled[n_train:n_train + SEQ_LEN + n_val],
        labels[n_train:n_train + SEQ_LEN + n_val],
        SEQ_LEN
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrackingLSTM(feat_dim).to(device)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    # Optimizer with gradient clipping
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    train_hist = {'total': [], 'pos': [], 'anomaly': [], 'metrics': []}
    val_hist = {'total': [], 'pos': [], 'anomaly': [], 'metrics': []}
    best_val_loss = float('inf')
    
    print(f"\nStarting training...")
    print(f"Data shape: {data_4d.shape}")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Training on device: {device}")
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        train_losses = {'total': 0., 'pos': 0., 'anomaly': 0.}
        train_metrics = []
        
        for seq, target, label in train_loader:
            seq = seq.to(device)
            target = target.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            pos_pred, anomaly_pred = model(seq)
            
            # Reshape target if needed
            target_flat = target.view(target.size(0), -1)
            
            # Calculate losses
            pos_loss = mse_loss(pos_pred, target_flat)
            anomaly_loss = bce_loss(anomaly_pred, label.unsqueeze(1))
            loss = pos_loss + 0.3 * anomaly_loss
            
            # Calculate metrics
            metrics = calculate_metrics(pos_pred, target_flat, anomaly_pred, label.unsqueeze(1))
            train_metrics.append(metrics)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            
            train_losses['total'] += loss.item()
            train_losses['pos'] += pos_loss.item()
            train_losses['anomaly'] += anomaly_loss.item()
        
        # Average training losses and metrics
        for k in train_losses:
            train_losses[k] /= len(train_loader)
        
        avg_train_metrics = {k: np.mean([m[k] for m in train_metrics]) for k in train_metrics[0]}
        train_hist['metrics'].append(avg_train_metrics)
        
        # Validation
        model.eval()
        val_losses = {'total': 0., 'pos': 0., 'anomaly': 0.}
        
        with torch.no_grad():
            for seq, target, label in val_loader:
                # Move to device
                seq = seq.to(device)
                target = target.to(device)
                label = label.to(device)
                
                # Forward pass
                pos_pred, anomaly_pred = model(seq)
                
                # Reshape target if needed
                target_flat = target.view(target.size(0), -1)
                
                # Calculate losses
                pos_loss = mse_loss(pos_pred, target_flat)
                anomaly_loss = bce_loss(anomaly_pred, label.unsqueeze(1))
                
                # Combine losses
                loss = pos_loss + 0.3 * anomaly_loss
                
                # Record losses
                val_losses['total'] += loss.item()
                val_losses['pos'] += pos_loss.item()
                val_losses['anomaly'] += anomaly_loss.item()
        
        # Average validation losses
        for k in val_losses:
            val_losses[k] /= len(val_loader)
            val_hist[k].append(val_losses[k])
        
        # Learning rate scheduling
        scheduler.step(val_losses['total'])
        current_lr = optimizer.param_groups[0]['lr']
        train_hist['lr'].append(current_lr)
        print(f"Learning rate: {current_lr}")
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'val_loss': best_val_loss,
            }, 'best_model.pth')
        
        # Print progress
        print(f"Epoch {epoch:02d}")
        print(f"Train | Total: {train_losses['total']:.4f}, Position: {train_losses['pos']:.4f}, "
              f"Anomaly: {train_losses['anomaly']:.4f}")
        print(f"Train Metrics | MAE: {avg_train_metrics['mae']:.4f}, RMSE: {avg_train_metrics['rmse']:.4f}, "
              f"F1: {avg_train_metrics['f1']:.4f}")
        print(f"Val   | Total: {val_losses['total']:.4f}, Position: {val_losses['pos']:.4f}, "
              f"Anomaly: {val_losses['anomaly']:.4f}\n")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(train_hist['total'], label='Train')
    plt.plot(val_hist['total'], label='Val')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(132)
    plt.plot(train_hist['pos'], label='Train')
    plt.plot(val_hist['pos'], label='Val')
    plt.title('Position Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(133)
    plt.plot(train_hist['anomaly'], label='Train')
    plt.plot(val_hist['anomaly'], label='Val')
    plt.title('Anomaly Detection Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("Training complete. Model saved as 'best_model.pth'")
    print("Training history plots saved as 'training_history.png'")

if __name__ == '__main__':
    main()