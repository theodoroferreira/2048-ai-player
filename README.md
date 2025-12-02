# 2048 AI Player

An AI-powered version of the classic 2048 game using Deep Q-Learning (DQN). Watch a neural network learn to play and master the game!

![2048 Game](meta/apple-touch-icon.png)

## Features

- Classic 2048 game implementation (based on [Gabriele Cirulli's original](https://github.com/gabrielecirulli/2048))
- Deep Q-Learning AI trained with TensorFlow
- Browser-based AI gameplay using ONNX Runtime Web
- Automated training pipeline
- Real-time Q-value visualization in console

## Quick Start

### Option 1: Train Your Own AI (Recommended)

**Windows:**
```bash
train_and_convert.bat
```

**Linux/Mac:**
```bash
chmod +x train_and_convert.sh
./train_and_convert.sh
```

This will:
1. Install Python dependencies
2. Train the AI for 100 episodes (~10-20 minutes)
3. Convert the model to ONNX format
4. Save everything to the `model/` directory

### Option 2: Manual Training

1. **Install dependencies:**
   ```bash
   pip install -r python/requirements.txt
   ```

2. **Train the model:**
   ```bash
   python python/train_dqn.py
   ```

3. **Convert to ONNX:**
   ```bash
   python python/convert_to_onnx.py
   ```

## Playing the Game

1. **Open the game:**
   - Simply open `index.html` in your web browser

2. **Manual play:**
   - Use arrow keys to play yourself

3. **Watch the AI:**
   - Click "Play AI" button or press 'A' key
   - Open browser console (F12) to see AI decision-making
   - Press 'A' again to stop

## How It Works

### Deep Q-Learning Architecture

The AI uses a Deep Q-Network (DQN) with the following architecture:

```
Input: 16 nodes (4x4 board with log2 encoding)
  ↓
Hidden Layer: 256 nodes (ReLU + Dropout)
  ↓
Hidden Layer: 256 nodes (ReLU + Dropout)
  ↓
Hidden Layer: 128 nodes (ReLU)
  ↓
Output: 4 nodes (Q-values for Up, Right, Down, Left)
```

### Training Process

1. **Experience Replay**: The agent stores past game experiences and learns from random samples
2. **Target Network**: Uses a separate network for stable Q-value estimation
3. **Epsilon-Greedy**: Balances exploration (trying new moves) vs exploitation (using learned moves)
4. **Reward Shaping**:
   - Rewards for merging tiles
   - Bonus for keeping cells empty
   - Penalties for invalid moves

### State Encoding

The board state is encoded using log2 transformation:
- Empty cell → 0
- Tile value 2 → 1
- Tile value 4 → 2
- Tile value 8 → 3
- And so on...

This creates a 16-dimensional vector representing the entire board.

## Project Structure

```
2048-ai-player/
├── index.html              # Main game interface
├── js/                     # JavaScript game engine
│   ├── game_manager.js     # Core game logic
│   ├── grid.js             # Board management
│   ├── trainer.js          # AI player (ONNX Runtime Web)
│   └── ...
├── python/                 # Training infrastructure
│   ├── game_2048.py        # Python game environment
│   ├── train_dqn.py        # DQN training script
│   ├── convert_to_onnx.py  # Model converter (Keras → ONNX)
│   ├── requirements.txt    # Python dependencies
│   └── README.md           # Detailed training docs
├── model/                  # Trained model (created after training)
│   └── model.onnx          # ONNX model file
└── style/                  # CSS styling
```

## Training Performance

After 1000 episodes of training, you can expect:
- **Average Score**: 1000-2000 points
- **Max Tile**: Commonly reaches 256-512
- **Best Runs**: Can achieve 1024 or even 2048 tile

For better performance, train for 2000-5000 episodes (see [python/README.md](python/README.md)).

## Advanced Usage

### Customizing Training

Edit `python/train_dqn.py` to adjust hyperparameters:

```python
agent = DQNAgent(
    learning_rate=0.001,    # Learning speed
    gamma=0.95,             # Discount factor
    epsilon_decay=0.995,    # Exploration decay rate
    memory_size=10000,      # Experience replay size
    batch_size=64           # Training batch size
)
```

### Evaluating Model Performance

Check training statistics:
```bash
cat python/models/training_stats.json
```

See detailed training guide: [python/README.md](python/README.md)

## Requirements

### For Training (Python)
- Python 3.10+
- TensorFlow 2.13-2.17
- NumPy <2.0
- tf2onnx, onnx

### For Playing (Browser)
- Modern web browser with JavaScript enabled
- ONNX Runtime Web (loaded automatically from CDN)

## Development

### CSS Development
```bash
gem install sass
sass --unix-newlines --watch style/main.scss
```

### Code Linting
```bash
jshint js/*.js
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Troubleshooting

### AI won't load
- Verify `model/model.json` exists
- Check browser console for errors
- Ensure you've trained and converted the model

### Training is slow
- Reduce episodes in `train_dqn.py`
- Use GPU acceleration (install `tensorflow-gpu`)
- Close other resource-intensive applications

### Poor AI performance
- Train for more episodes (2000+)
- Adjust reward function in `game_2048.py`
- Tune hyperparameters in `train_dqn.py`

## Credits

- Original 2048 game by [Gabriele Cirulli](https://github.com/gabrielecirulli/2048)
- Based on 1024 by Veewo Studio
- Conceptually similar to Threes by Asher Vollmer
- AI implementation using Deep Q-Learning

## License

See [LICENSE](LICENSE) file for details.