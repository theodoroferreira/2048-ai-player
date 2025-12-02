"""
Maximum Performance Deep Q-Learning training for 2048.
Optimized to fully utilize GPU and CPU resources.
"""

import random
import json
import os
from collections import deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # Pre-allocate all GPU memory

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from game_2048 import Game2048, choose_action_heuristic

# Configure GPU for maximum performance
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Pre-allocate all GPU memory for faster access
            tf.config.experimental.set_memory_growth(gpu, False)
        print(f"✓ GPU configured for maximum performance")
    except RuntimeError as e:
        print(f"GPU configuration: {e}")

print(f"TensorFlow using {len(tf.config.list_physical_devices('GPU'))} GPU(s)")


class DQNAgent:
    """Deep Q-Network agent - optimized for batch processing."""

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
        batch_size: int = 256  # Larger batches = better GPU utilization
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Pre-compile model with sample data for faster first run
        dummy_input = np.random.random((1, self.state_size)).astype(np.float32)
        self.model.predict(dummy_input, verbose=0)
        print("✓ Model warmed up")

    def _build_model(self) -> keras.Model:
        """Build neural network - optimized for GPU."""
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @tf.function  # Graph mode for faster execution
    def _predict_batch(self, states):
        return self.model(states, training=False)

    def act(self, state, env=None, valid_actions=None) -> int:
        """
        Choose an action using epsilon-greedy policy with heuristic fallback.

        Args:
            state: Current state
            env: Game environment (optional, for heuristic)
            valid_actions: List of valid actions (optional)
        """
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        # Exploration: use heuristic if env is provided, otherwise random
        if np.random.random() <= self.epsilon:
            if env is not None:
                return choose_action_heuristic(env)
            else:
                return random.choice(valid_actions)

        # Exploitation: use neural network
        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)[0]

        valid_q_values = [(action, q_values[action]) for action in valid_actions]
        return max(valid_q_values, key=lambda x: x[1])[0]

    def replay_batch(self, num_batches=3):
        """Train on multiple batches at once for better GPU utilization."""
        if len(self.memory) < self.batch_size:
            return

        for _ in range(num_batches):
            minibatch = random.sample(self.memory, self.batch_size)

            states = np.array([exp[0] for exp in minibatch], dtype=np.float32)
            actions = np.array([exp[1] for exp in minibatch])
            rewards = np.array([exp[2] for exp in minibatch], dtype=np.float32)
            next_states = np.array([exp[3] for exp in minibatch], dtype=np.float32)
            dones = np.array([exp[4] for exp in minibatch])

            # Batch predictions - GPU shines here
            current_q_values = self.model.predict(states, verbose=0, batch_size=self.batch_size)
            next_q_values = self.target_model.predict(next_states, verbose=0, batch_size=self.batch_size)

            # Vectorized target calculation
            targets = current_q_values.copy()
            for i in range(self.batch_size):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

            # Train in one big batch
            self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath: str):
        self.model.save(filepath)
        print(f"Model saved to {filepath}")


def play_episode_worker(env, agent, max_steps):
    """Worker function for parallel episode execution with heuristic guidance."""
    state = env.reset()
    total_reward = 0
    experiences = []

    for step in range(max_steps):
        # Pass environment to act() so it can use heuristic during exploration
        action = agent.act(state, env=env)
        next_state, reward, done = env.step(action)
        total_reward += reward

        experiences.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            break

    return experiences, env.score, env.get_max_tile()


def train(
    episodes: int = 100,
    max_steps: int = 150,  # Reduced slightly for speed
    target_update_freq: int = 5,  # Update more frequently
    parallel_envs: int = 4,  # Run 4 games in parallel
    model_dir: str = "python/models"
):
    """Train with parallel environments for maximum resource usage."""
    os.makedirs(model_dir, exist_ok=True)

    # Create multiple environments for parallel execution
    envs = [Game2048() for _ in range(parallel_envs)]
    agent = DQNAgent()

    scores = []
    max_tiles = []
    epsilons = []

    print(f"Starting MAXIMUM PERFORMANCE training...")
    print(f"Episodes: {episodes}, Parallel environments: {parallel_envs}")
    print(f"Max steps: {max_steps}, Batch size: {agent.batch_size}")
    print("-" * 60)

    import time
    start_time = time.time()

    episode = 0
    with ThreadPoolExecutor(max_workers=parallel_envs) as executor:
        while episode < episodes:
            # Run multiple episodes in parallel
            futures = []
            for env in envs[:min(parallel_envs, episodes - episode)]:
                future = executor.submit(play_episode_worker, env, agent, max_steps)
                futures.append(future)

            # Collect results
            for future in futures:
                if episode >= episodes:
                    break

                experiences, score, max_tile = future.result()

                # Store all experiences
                for exp in experiences:
                    agent.remember(*exp)

                scores.append(score)
                max_tiles.append(max_tile)
                epsilons.append(agent.epsilon)

                # Train on multiple batches for better GPU utilization
                if len(agent.memory) >= agent.batch_size:
                    agent.replay_batch(num_batches=3)

                episode += 1

                # Progress updates
                if episode % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_score = np.mean(scores[-10:])
                    avg_max_tile = np.mean(max_tiles[-10:])
                    eps_per_min = episode / (elapsed / 60)
                    print(f"Episode {episode}/{episodes} | "
                          f"Score: {avg_score:.0f} | "
                          f"Tile: {avg_max_tile:.0f} | "
                          f"ε: {agent.epsilon:.3f} | "
                          f"Speed: {eps_per_min:.1f} ep/min")

                # Update target network
                if episode % target_update_freq == 0:
                    agent.update_target_model()

    # Save final model
    final_model_path = os.path.join(model_dir, "model_final.h5")
    agent.save(final_model_path)

    # Save statistics
    stats = {
        "episodes": episodes,
        "scores": [int(s) for s in scores],
        "max_tiles": [int(t) for t in max_tiles],
        "epsilons": [float(e) for e in epsilons],
        "final_avg_score": float(np.mean(scores[-100:])),
        "final_avg_max_tile": float(np.mean(max_tiles[-100:])),
        "best_score": int(max(scores)),
        "best_tile": int(max(max_tiles)),
        "training_time_minutes": (time.time() - start_time) / 60,
        "parallel_envs": parallel_envs
    }

    stats_path = os.path.join(model_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✓ Training completed!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average speed: {episodes/(total_time/60):.1f} episodes/minute")
    print(f"Final avg score: {stats['final_avg_score']:.0f}")
    print(f"Final avg max tile: {stats['final_avg_max_tile']:.0f}")
    print(f"Best score: {stats['best_score']}")
    print(f"Best tile: {stats['best_tile']}")
    print("=" * 60)


if __name__ == "__main__":
    train(
        episodes=100,
        max_steps=150,
        parallel_envs=4,  # Adjust based on CPU cores (Colab has ~2 cores)
        target_update_freq=5
    )
