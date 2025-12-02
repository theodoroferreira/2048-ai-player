import numpy as np
from typing import Tuple, Optional


class Game2048:
    """
    2048 game environment for reinforcement learning.
    Matches the JavaScript implementation's logic.
    """

    def __init__(self, size: int = 4):
        self.size = size
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the game to initial state with 2 random tiles."""
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.game_over = False

        # Add two starting tiles
        self._add_random_tile()
        self._add_random_tile()

        return self.get_state()

    def _add_random_tile(self) -> None:
        """Add a random tile (2 with 90% probability, 4 with 10%) to an empty cell."""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = empty_cells[np.random.randint(len(empty_cells))]
            self.board[x, y] = 4 if np.random.random() < 0.1 else 2

    def get_state(self) -> np.ndarray:
        """
        Get the current board state as a normalized vector.
        Uses log2 encoding: 0->0, 2->1, 4->2, 8->3, etc.
        Returns flattened 16-element array.
        """
        state = np.zeros(self.size * self.size, dtype=np.float32)
        flat_board = self.board.flatten()
        for i, val in enumerate(flat_board):
            if val > 0:
                state[i] = np.log2(val)
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute an action.

        Args:
            action: 0=up, 1=right, 2=down, 3=left

        Returns:
            (new_state, reward, done)
        """
        if self.game_over:
            return self.get_state(), 0.0, True

        old_board = self.board.copy()
        old_score = self.score

        # Execute move
        moved = self._move(action)

        # Calculate reward
        if not moved:
            # Invalid move penalty
            reward = -10.0
        else:
            # Reward is the score gained from merges
            score_gain = self.score - old_score
            reward = float(score_gain)

            # Add small bonus for keeping cells empty
            empty_cells = np.sum(self.board == 0)
            reward += empty_cells * 0.1

            # Add new tile after valid move
            self._add_random_tile()

            # Check if game is over
            if not self._has_valid_moves():
                self.game_over = True

        return self.get_state(), reward, self.game_over

    def _move(self, direction: int) -> bool:
        """
        Perform a move in the given direction.
        Returns True if the board changed, False otherwise.
        """
        old_board = self.board.copy()

        if direction == 0:  # Up
            self.board = self._move_up()
        elif direction == 1:  # Right
            self.board = self._move_right()
        elif direction == 2:  # Down
            self.board = self._move_down()
        elif direction == 3:  # Left
            self.board = self._move_left()

        return not np.array_equal(old_board, self.board)

    def _move_left(self) -> np.ndarray:
        """Move and merge tiles to the left."""
        new_board = np.zeros_like(self.board)

        for i in range(self.size):
            # Get non-zero values in the row
            row = self.board[i, :]
            non_zero = row[row != 0]

            # Merge adjacent equal tiles
            merged = []
            skip = False
            for j in range(len(non_zero)):
                if skip:
                    skip = False
                    continue

                if j + 1 < len(non_zero) and non_zero[j] == non_zero[j + 1]:
                    merged_value = non_zero[j] * 2
                    merged.append(merged_value)
                    self.score += merged_value
                    skip = True
                else:
                    merged.append(non_zero[j])

            # Place merged values in the new board
            new_board[i, :len(merged)] = merged

        return new_board

    def _move_right(self) -> np.ndarray:
        """Move and merge tiles to the right."""
        flipped = np.fliplr(self.board)
        self.board = flipped
        moved = self._move_left()
        return np.fliplr(moved)

    def _move_up(self) -> np.ndarray:
        """Move and merge tiles up."""
        transposed = self.board.T
        self.board = transposed
        moved = self._move_left()
        return moved.T

    def _move_down(self) -> np.ndarray:
        """Move and merge tiles down."""
        transposed = self.board.T
        flipped = np.fliplr(transposed)
        self.board = flipped
        moved = self._move_left()
        return np.fliplr(moved).T

    def _has_valid_moves(self) -> bool:
        """Check if any valid moves are available."""
        # Check for empty cells
        if np.any(self.board == 0):
            return True

        # Check for possible merges horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j + 1]:
                    return True

        # Check for possible merges vertically
        for i in range(self.size - 1):
            for j in range(self.size):
                if self.board[i, j] == self.board[i + 1, j]:
                    return True

        return False

    def get_max_tile(self) -> int:
        """Return the maximum tile value on the board."""
        return int(np.max(self.board))

    def clone(self) -> 'Game2048':
        """Create a copy of the current game state."""
        new_game = Game2048(self.size)
        new_game.board = self.board.copy()
        new_game.score = self.score
        new_game.game_over = self.game_over
        return new_game
