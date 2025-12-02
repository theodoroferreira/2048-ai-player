# 2048 AI Training with Deep Q-Learning

This directory contains the training infrastructure for teaching an AI to play 2048 using Deep Q-Learning (DQN).

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   ```

## Training the Model

### Quick Start (Fast Training - 500 episodes)

For testing and quick results:

```bash
python python/train_dqn.py
```

This will train for 1000 episodes (takes ~20-40 minutes depending on hardware).

### Custom Training

Edit [train_dqn.py](train_dqn.py) and modify the training parameters:

```python
train(
    episodes=2000,        # More episodes = better performance
    max_steps=2000,       # Maximum moves per game
    target_update_freq=10,# Update target network frequency
    save_freq=100         # Save model every N episodes
)
```

### Training Output

The training script will:
- Print progress every 10 episodes
- Save models every 100 episodes to `python/models/`
- Save the final model as `python/models/model_final.h5`
- Save training statistics to `python/models/training_stats.json`

Example output:
```
Episode 10/1000 | Avg Score: 523.2 | Avg Max Tile: 128.0 | Epsilon: 0.904
Episode 20/1000 | Avg Score: 741.5 | Avg Max Tile: 256.0 | Epsilon: 0.817
...
```

## Converting Model to ONNX

After training, convert the model for browser use:

```bash
python python/convert_to_onnx.py
```

This will:
- Load the trained Keras model from `python/models/model_final.h5`
- Convert it to ONNX format
- Save to `model/model.onnx` (used by the web interface)

### Custom Conversion

```bash
python python/convert_to_onnx.py --input python/models/model_ep100.h5 --output model/model.onnx
```

## Testing the AI in the Browser

1. **Make sure the model is converted:**
   ```bash
   ls model/model.onnx  # Should exist
   ```

2. **Open the game:**
   - Open `index.html` in a web browser
   - Click the "Play AI" button or press 'A' key
   - Watch the AI play!

3. **Console output:**
   Open browser DevTools (F12) to see:
   - Move-by-move decisions
   - Q-values for each action
   - Game statistics

## Understanding the Results

### Training Metrics

- **Score**: Points earned in the game (higher is better)
- **Max Tile**: Highest tile achieved (e.g., 256, 512, 1024, 2048)
- **Epsilon**: Exploration rate (starts at 1.0, decays to 0.01)

### Expected Performance

After 1000 episodes:
- Average Score: 1000-2000
- Average Max Tile: 256-512
- Best runs: Can reach 1024 or even 2048 tile

For better performance, train for 2000-5000 episodes.

## Architecture

### Deep Q-Network (DQN)

The model architecture:
```
Input Layer:  16 nodes (4x4 board, log2 encoded)
Hidden Layer: 256 nodes (ReLU + Dropout 0.2)
Hidden Layer: 256 nodes (ReLU + Dropout 0.2)
Hidden Layer: 128 nodes (ReLU)
Output Layer: 4 nodes (Q-values for Up, Right, Down, Left)
```

### Key Features

1. **Experience Replay**: Stores past experiences and samples randomly for training
2. **Target Network**: Separate network for stable Q-value estimation
3. **Epsilon-Greedy**: Balances exploration (random moves) vs exploitation (best known moves)
4. **Reward Shaping**:
   - Positive reward for merging tiles (score gained)
   - Small bonus for keeping cells empty
   - Penalty for invalid moves

## Troubleshooting

### Training is slow
- Reduce `episodes` or `max_steps`
- Use GPU acceleration (requires `tensorflow-gpu`)

### Model performs poorly
- Train for more episodes (2000-5000)
- Adjust hyperparameters in [train_dqn.py](train_dqn.py)
- Check that reward function matches game mechanics

### Browser can't load model
- Verify `model/model.onnx` exists
- Check browser console for errors
- Ensure ONNX Runtime Web is loaded

### Memory issues during training
- Reduce `memory_size` in DQNAgent
- Reduce `batch_size`
- Close other applications

## Advanced Usage

### Evaluating a Trained Model

Create a test script to evaluate performance:

```python
from game_2048 import Game2048
from train_dqn import DQNAgent
import numpy as np

# Load trained agent
agent = DQNAgent()
agent.load("python/models/model_final.h5")
agent.epsilon = 0.0  # No exploration

# Play games
env = Game2048()
scores = []

for i in range(100):
    state = env.reset()
    done = False

    while not done:
        action = agent.act(state)
        state, reward, done = env.step(action)

    scores.append(env.score)
    print(f"Game {i+1}: Score={env.score}, Max Tile={env.get_max_tile()}")

print(f"\nAverage Score: {np.mean(scores):.1f}")
```

### Hyperparameter Tuning

Key parameters to experiment with:
- `learning_rate`: 0.0001 - 0.01 (default: 0.001)
- `gamma`: 0.9 - 0.99 (default: 0.95)
- `epsilon_decay`: 0.99 - 0.999 (default: 0.995)
- Network architecture (layer sizes, dropout rates)

## Files

- `game_2048.py`: 2048 game environment implementation
- `train_dqn.py`: Deep Q-Learning training script
- `convert_to_onnx.py`: Model conversion script (Keras to ONNX)
- `requirements.txt`: Python dependencies
- `models/`: Directory for saved models (created during training)

## References

- [Deep Q-Learning Paper](https://arxiv.org/abs/1312.5602)
- [ONNX Runtime Web Documentation](https://onnxruntime.ai/docs/tutorials/web/)
- [Original 2048 Game](https://github.com/gabrielecirulli/2048)
