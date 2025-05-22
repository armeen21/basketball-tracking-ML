import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import joblib
from sklearn.preprocessing import MinMaxScaler
from SampleModel import TrackingLSTM, stream_and_parse, TrackingDataset

# Configuration
BATCH_SIZE = 64
CONTINUE_EPOCHS = 20
VISUALIZATION_SAMPLES = 5
ANIMATION_FRAMES = 50
HIDDEN_SIZE = 256
NUM_LAYERS = 3
SEQ_LEN = 30
INJURY_THRESH = 50.0
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def load_model_and_scaler():
    """Load the trained model and scaler."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try loading the continued model first, fall back to best model
    try:
        print("Attempting to load continued model...")
        checkpoint = torch.load('continued_model.pth', map_location=device)
        feat_dim = checkpoint['model_state_dict']['encoder.0.weight'].size(1)
        model = TrackingLSTM(feat_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = checkpoint['scaler']
        print(f"Loaded continued model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.6f}")
    except (FileNotFoundError, KeyError):
        print("No continued model found, loading best model...")
        checkpoint = torch.load('best_model.pth', map_location=device)
        feat_dim = checkpoint['model_state_dict']['encoder.0.weight'].size(1)
        model = TrackingLSTM(feat_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = checkpoint['scaler']
        print("Loaded best model")
    
    return model, scaler, device

def visualize_predictions(model, data_loader, scaler, device, num_samples=5):
    """Create static visualizations of predictions vs actual trajectories."""
    model.eval()
    
    # Get some sample sequences and predictions
    sequences = []
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for seq, target, _ in data_loader:
            if len(sequences) >= num_samples:
                break
                
            seq = seq.to(device)
            target = target.to(device)
            
            pos_pred, _ = model(seq)
            
            # Reshape predictions and targets back to 4D
            B, T = seq.shape[:2]
            P, J = seq.shape[2:4] if seq.dim() > 3 else (1, 1)
            C = 3  # x, y, z coordinates
            
            # Convert to numpy and inverse transform
            seq_np = seq[0].cpu().numpy().reshape(T, -1)
            pred_np = pos_pred[0].cpu().numpy().reshape(1, -1)
            target_np = target[0].cpu().numpy().reshape(1, -1)
            
            # Inverse transform
            seq_np = scaler.inverse_transform(seq_np).reshape(T, P, J, C)
            pred_np = scaler.inverse_transform(pred_np).reshape(P, J, C)
            target_np = scaler.inverse_transform(target_np).reshape(P, J, C)
            
            sequences.append(seq_np)
            predictions.append(pred_np)
            actuals.append(target_np)
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 5 * num_samples))
    
    for i in range(num_samples):
        seq = sequences[i]
        pred = predictions[i]
        actual = actuals[i]
        
        # 3D trajectory plot for all entities
        ax1 = fig.add_subplot(num_samples, 3, 3*i + 1, projection='3d')
        
        # Plot sequence for each player/joint
        for p in range(seq.shape[1]):  # players
            for j in range(seq.shape[2]):  # joints
                trajectory = seq[:, p, j]
                ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                        '-', label=f'P{p}J{j} History', alpha=0.5)
                ax1.scatter(*pred[p, j], marker='*', s=100, label=f'P{p}J{j} Pred')
                ax1.scatter(*actual[p, j], marker='o', s=100, label=f'P{p}J{j} Act')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'Sample {i+1}: 3D Trajectories')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Top-down view (XY plane)
        ax2 = fig.add_subplot(num_samples, 3, 3*i + 2)
        for p in range(seq.shape[1]):
            for j in range(seq.shape[2]):
                trajectory = seq[:, p, j]
                ax2.plot(trajectory[:, 0], trajectory[:, 1], '-', alpha=0.5)
                ax2.scatter(pred[p, j, 0], pred[p, j, 1], marker='*', s=100)
                ax2.scatter(actual[p, j, 0], actual[p, j, 1], marker='o', s=100)
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title(f'Sample {i+1}: Top-down View')
        
        # Side view (XZ plane)
        ax3 = fig.add_subplot(num_samples, 3, 3*i + 3)
        for p in range(seq.shape[1]):
            for j in range(seq.shape[2]):
                trajectory = seq[:, p, j]
                ax3.plot(trajectory[:, 0], trajectory[:, 2], '-', alpha=0.5)
                ax3.scatter(pred[p, j, 0], pred[p, j, 2], marker='*', s=100)
                ax3.scatter(actual[p, j, 0], actual[p, j, 2], marker='o', s=100)
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title(f'Sample {i+1}: Side View')
    
    plt.tight_layout()
    plt.savefig('trajectory_predictions.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_animation(model, data_loader, scaler, device):
    """Create an animated visualization of predictions."""
    model.eval()
    
    # Get a single sequence for animation
    for seq, target, _ in data_loader:
        seq = seq.to(device)
        target = target.to(device)
        break
    
    with torch.no_grad():
        pos_pred, _ = model(seq)
    
    # Reshape and convert to numpy
    B, T = seq.shape[:2]
    P, J = seq.shape[2:4] if seq.dim() > 3 else (1, 1)
    C = 3
    
    seq_np = seq[0].cpu().numpy().reshape(T, -1)
    pred_np = pos_pred[0].cpu().numpy().reshape(1, -1)
    target_np = target[0].cpu().numpy().reshape(1, -1)
    
    # Inverse transform
    seq_np = scaler.inverse_transform(seq_np).reshape(T, P, J, C)
    pred_np = scaler.inverse_transform(pred_np).reshape(P, J, C)
    target_np = scaler.inverse_transform(target_np).reshape(P, J, C)
    
    # Create animation
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    def update(frame):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        
        # 3D plot
        for p in range(P):
            for j in range(J):
                trajectory = seq_np[:frame, p, j]
                if frame > 0:
                    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                            '-', alpha=0.5, label=f'P{p}J{j}')
                if frame == len(seq_np):
                    ax1.scatter(*pred_np[p, j], marker='*', s=100, label=f'P{p}J{j} Pred')
                    ax1.scatter(*target_np[p, j], marker='o', s=100, label=f'P{p}J{j} Act')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Trajectories')
        if frame == len(seq_np):
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Top-down view
        for p in range(P):
            for j in range(J):
                trajectory = seq_np[:frame, p, j]
                if frame > 0:
                    ax2.plot(trajectory[:, 0], trajectory[:, 1], '-', alpha=0.5)
                if frame == len(seq_np):
                    ax2.scatter(pred_np[p, j, 0], pred_np[p, j, 1], marker='*', s=100)
                    ax2.scatter(target_np[p, j, 0], target_np[p, j, 1], marker='o', s=100)
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Top-down View')
        
        # Side view
        for p in range(P):
            for j in range(J):
                trajectory = seq_np[:frame, p, j]
                if frame > 0:
                    ax3.plot(trajectory[:, 0], trajectory[:, 2], '-', alpha=0.5)
                if frame == len(seq_np):
                    ax3.scatter(pred_np[p, j, 0], pred_np[p, j, 2], marker='*', s=100)
                    ax3.scatter(target_np[p, j, 0], target_np[p, j, 2], marker='o', s=100)
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('Side View')
    
    anim = FuncAnimation(fig, update, frames=len(seq_np)+1, interval=50)
    anim.save('trajectory_animation.gif', writer='pillow', dpi=150)
    plt.close()

def continue_training(model, train_loader, val_loader, device, num_epochs=20):
    """Continue training the model."""
    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_losses = {'total': 0., 'pos': 0., 'anomaly': 0.}
        
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
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            train_losses['total'] += loss.item()
            train_losses['pos'] += pos_loss.item()
            train_losses['anomaly'] += anomaly_loss.item()
        
        # Average training losses
        for k in train_losses:
            train_losses[k] /= len(train_loader)
        
        # Validation
        model.eval()
        val_losses = {'total': 0., 'pos': 0., 'anomaly': 0.}
        
        with torch.no_grad():
            for seq, target, label in val_loader:
                seq = seq.to(device)
                target = target.to(device)
                label = label.to(device)
                
                pos_pred, anomaly_pred = model(seq)
                
                # Reshape target if needed
                target_flat = target.view(target.size(0), -1)
                
                # Calculate losses
                pos_loss = mse_loss(pos_pred, target_flat)
                anomaly_loss = bce_loss(anomaly_pred, label.unsqueeze(1))
                loss = pos_loss + 0.3 * anomaly_loss
                
                val_losses['total'] += loss.item()
                val_losses['pos'] += pos_loss.item()
                val_losses['anomaly'] += anomaly_loss.item()
        
        # Average validation losses
        for k in val_losses:
            val_losses[k] /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_losses['total'])
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'continued_model.pth')
        
        print(f"Epoch {epoch:02d}")
        print(f"Train | Total: {train_losses['total']:.4f}, Position: {train_losses['pos']:.4f}, "
              f"Anomaly: {train_losses['anomaly']:.4f}")
        print(f"Val   | Total: {val_losses['total']:.4f}, Position: {val_losses['pos']:.4f}, "
              f"Anomaly: {val_losses['anomaly']:.4f}\n")

def main():
    # Load model and scaler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, scaler, device = load_model_and_scaler()
    
    # Load and preprocess data
    print("Loading data...")
    data = stream_and_parse('sampletrackingdata.json')
    N, P, J, C = data.shape
    feat_dim = P * J * C
    
    # Split indices
    num_windows = N - SEQ_LEN
    n_test = int(num_windows * TEST_RATIO)
    n_val = int(num_windows * VAL_RATIO)
    n_train = num_windows - n_val - n_test
    
    # Create labels
    accel = data[2:] - 2 * data[1:-1] + data[:-2]  # (N-2, P, J, 3)
    mags = np.linalg.norm(accel, axis=3)  # (N-2, P, J)
    max_mags = mags.max(axis=(1,2))  # (N-2,)
    labels = (max_mags > INJURY_THRESH).astype(np.float32)
    labels = np.concatenate(([0, 0], labels))  # (N,)
    
    # Scale data
    data_scaled = scaler.transform(data.reshape(N, -1))
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
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_predictions(model, val_loader, scaler, device, num_samples=VISUALIZATION_SAMPLES)
    create_animation(model, val_loader, scaler, device)
    
    print("\nDone! Check 'trajectory_predictions.png' and 'trajectory_animation.gif' for visualizations.")

if __name__ == '__main__':
    main() 