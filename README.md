# Basketball Player and Ball Tracking ML Model

A sophisticated machine learning model for tracking basketball players and ball movements, with built-in injury risk assessment capabilities. The model uses LSTM architecture to predict player/ball positions and detect potential injury-risk scenarios in real-time.

## Features

- **Multi-Entity Tracking**
  - Simultaneous tracking of multiple players and the ball
  - Support for individual joint tracking per player
  - 3D position prediction and trajectory analysis

- **Advanced Architecture**
  - 3-layer LSTM with 256 hidden units
  - Sophisticated encoder-decoder architecture
  - Built-in anomaly detection for injury risk assessment
  - Dropout and gradient clipping for robust training

- **Comprehensive Analytics**
  - Per-frame position predictions
  - Real-time injury risk scoring
  - Five-second interval risk aggregates
  - Court heatmap generation
  - Performance metrics (RMSE, precision, recall, F1)
  - JSON output format for dashboard integration

## Requirements

```
torch==2.2.0
numpy==1.24.3
matplotlib==3.8.2
scikit-learn==1.3.2
ijson==3.2.3
joblib==1.3.2
pandas==1.5.3
tqdm==4.66.1
pillow==10.2.0
tensorboard==2.11.2
```

## Data Requirements

This project requires the following data files that are not included in the repository due to size constraints:

1. `sampletrackingdata.json` - Main tracking data file
   - Format: JSON/JSONL with player and ball positions
   - Required location: root directory
   - Size: ~1.8GB

2. Model files:
   - `best_model.pth` - Best performing model checkpoint
   - `continued_model.pth` - Latest model checkpoint

To obtain these files:
1. Contact the repository owner
2. Generate your own data following the format specified in the documentation
3. Train your own model using the provided scripts

### Data Format

The tracking data should follow this structure:
```json
{
  "samples": {
    "players": [
      {
        "joints": {
          "joint_name": [x, y, z],
          ...
        }
      }
    ],
    "ball": [
      {
        "pos": [x, y, z]
      }
    ]
  }
}
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/armeen21/basketball-tracking-ml.git
cd basketball-tracking-ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Obtain the required data files and place them in the appropriate locations

## Usage

### Training the Model

```python
python SampleModel.py
```

The model will automatically:
- Load and preprocess the tracking data
- Split into training/validation/test sets
- Train the LSTM model
- Save the best model and training history
- Generate performance plots

### Visualization

```python
python VisualizePredictions.py
```

This will generate:
- Trajectory predictions visualization
- Animated movement patterns
- Performance metrics plots

