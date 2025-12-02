"""
Deep Q-Learning training script for 2048 game.
Uses experience replay and target network for stable training.
"""

import random
import json
import os
from collections import deque
import numpy as np

# 1. Set the environment variable BEFORE importing TensorFlow/Keras
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Now import TensorFlow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from game_2048 import Game2048, choose_action_heuristic

# Verify suppression
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class DQNAgent:
    """Deep Q-Network agent for playing 2048."""

    def __init__(
        self,
        state_size: int = 16,
        action_size: int = 4,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000,
        batch_size: int = 256
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Experience replay memory
        self.memory = deque(maxlen=memory_size)

        # Main model (for action selection)
        self.model = self._build_model()

        # Target model (for Q-value calculation)
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self) -> keras.Model:
        """Build the neural network model."""
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_size, activation='linear')  # Q-values for each action
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            jit_compile=True
        )

        return model

    def update_target_model(self):
        """Copy weights from main model to target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, env=None, valid_actions=None) -> int:
        """
        Choose an action using epsilon-greedy policy with heuristic guidance.

        Args:
            state: Current game state
            env: Game environment (optional, for heuristic)
            valid_actions: List of valid action indices (optional)

        Returns:
            Selected action (0-3)
        """
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        # Exploration: use heuristic if env is provided, otherwise random
        if np.random.random() <= self.epsilon:
            if env is not None:
                return choose_action_heuristic(env)
            else:
                return random.choice(valid_actions)

        # Exploitation: best action according to model
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)[0]

        # Choose best action among valid ones
        valid_q_values = [(action, q_values[action]) for action in valid_actions]
        return max(valid_q_values, key=lambda x: x[1])[0]

    def replay(self):
        """Train the model using a random batch from memory."""
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Predict Q-values for starting state
        current_q_values = self.model.predict(states, verbose=0)

        # Predict Q-values for next state using target model
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Update Q-values using Bellman equation
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train the model
        self.model.fit(states, current_q_values, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath: str):
        """Save the model."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load a saved model."""
        self.model = keras.models.load_model(filepath)
        self.update_target_model()
        print(f"Model loaded from {filepath}")


def train(
    episodes: int = 100,
    max_steps: int = 200,
    target_update_freq: int = 10,
    batch_size: int = 256,
    model_dir: str = "python/models"
):
    """
    Train the DQN agent to play 2048.

    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        target_update_freq: Update target network every N episodes
        batch_size: Batch size for training
        model_dir: Directory to save models
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Initialize environment and agent
    env = Game2048()
    agent = DQNAgent()

    # Training statistics
    scores = []
    max_tiles = []
    epsilons = []

    print("Starting training...")
    print(f"Episodes: {episodes}, Max steps per episode: {max_steps}")
    print("-" * 60)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # Choose action (with heuristic guidance during exploration)
            action = agent.act(state, env=env)

            # Take action
            next_state, reward, done = env.step(action)
            total_reward += reward

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train the model
            agent.replay()

            state = next_state

            if done:
                break

        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_model()

        # Record statistics
        scores.append(env.score)
        max_tiles.append(env.get_max_tile())
        epsilons.append(agent.epsilon)

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            avg_max_tile = np.mean(max_tiles[-10:])
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Score: {avg_score:.1f} | "
                  f"Avg Max Tile: {avg_max_tile:.0f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    # Save final model
    final_model_path = os.path.join(model_dir, "model_final.h5")
    agent.save(final_model_path)

    # Save training statistics
    stats = {
        "episodes": episodes,
        "scores": [int(s) for s in scores],
        "max_tiles": [int(t) for t in max_tiles],
        "epsilons": [float(e) for e in epsilons],
        "final_avg_score": float(np.mean(scores[-100:])),
        "final_avg_max_tile": float(np.mean(max_tiles[-100:])),
        "best_score": int(max(scores)),
        "best_tile": int(max(max_tiles))
    }

    stats_path = os.path.join(model_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Final average score (last 100 episodes): {stats['final_avg_score']:.1f}")
    print(f"Final average max tile (last 100 episodes): {stats['final_avg_max_tile']:.0f}")
    print(f"Best score achieved: {stats['best_score']}")
    print(f"Best tile achieved: {stats['best_tile']}")
    print(f"Model saved to: {final_model_path}")
    print(f"Statistics saved to: {stats_path}")
    print("=" * 60)


if __name__ == "__main__":
    # Train the model
    train(
        episodes=100,
        max_steps=200,
        target_update_freq=10
    )
