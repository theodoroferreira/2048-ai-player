import numpy as np
from dataclasses import dataclass, field


@dataclass
class Game2048:
    size: int = 4
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    board: np.ndarray = field(init=False)
    score: int = field(init=False)

    def __post_init__(self):
        self.reset()

    # -------------------------------------------------------
    # API principal: reset / step / can_move
    # -------------------------------------------------------
    def reset(self):
        """Reinicia o jogo: tabuleiro vazio + duas peças iniciais."""
        self.board = np.zeros((self.size, self.size), dtype=np.int64)
        self.score = 0
        self._add_random_tile()
        self._add_random_tile()
        return self.board.copy()

    def step(self, action: int):
        """
        Executa uma ação no ambiente.
        action: 0=cima, 1=direita, 2=baixo, 3=esquerda
        Retorna: (novo_board, reward, done)
        """
        if action not in (0, 1, 2, 3):
            raise ValueError("Ação inválida. Use 0=cima, 1=dir, 2=baixo, 3=esq.")

        old_board = self.board.copy()
        new_board, score_gain = self._move(old_board, action)

        moved = not np.array_equal(old_board, new_board)
        reward = float(score_gain)

        if moved:
            self.board = new_board
            self.score += score_gain
            self._add_random_tile()

        done = not self.can_move()
        return self.board.copy(), reward, done

    def can_move(self) -> bool:
        """Retorna True se ainda há movimentos possíveis."""
        board = self.board
        # Se existe célula vazia, ainda pode jogar
        if np.any(board == 0):
            return True

        # Verifica se existe par adjacente igual (horizontal ou vertical)
        for x in range(self.size):
            for y in range(self.size):
                v = board[x, y]
                if x + 1 < self.size and board[x + 1, y] == v:
                    return True
                if y + 1 < self.size and board[x, y + 1] == v:
                    return True

        return False

    # -------------------------------------------------------
    # Lógica interna do jogo
    # -------------------------------------------------------
    def _add_random_tile(self):
        """Adiciona uma peça 2 (90%) ou 4 (10%) em uma posição vazia."""
        empty_positions = np.argwhere(self.board == 0)
        if empty_positions.size == 0:
            return
        idx = self.rng.integers(len(empty_positions))
        x, y = empty_positions[idx]
        value = 4 if self.rng.random() < 0.1 else 2
        self.board[x, y] = value

    def _move(self, board: np.ndarray, action: int):
        """
        Aplica o movimento na cópia do tabuleiro.
        action: 0=cima, 1=dir, 2=baixo, 3=esq
        Retorna: (novo_board, score_ganho)
        """
        if action == 3:   # esquerda
            new_board, score_gain = self._move_left(board)
        elif action == 1: # direita
            flipped = np.fliplr(board)
            moved_board, score_gain = self._move_left(flipped)
            new_board = np.fliplr(moved_board)
        elif action == 0: # cima
            transposed = board.T
            moved_board, score_gain = self._move_left(transposed)
            new_board = moved_board.T
        elif action == 2: # baixo
            transposed = board.T
            flipped = np.fliplr(transposed)
            moved_board, score_gain = self._move_left(flipped)
            new_board = np.fliplr(moved_board).T
        else:
            raise ValueError("Ação inválida.")

        return new_board, score_gain

    def _move_left(self, board: np.ndarray):
        """
        Move todas as linhas para a esquerda (lógica base).
        Retorna: (novo_board, score_ganho)
        """
        size = self.size
        new_board = np.zeros_like(board)
        score_gain = 0

        for i in range(size):
            row = board[i, :]
            # Remove zeros
            non_zero = row[row != 0]
            merged_row = []
            j = 0
            while j < len(non_zero):
                if (j + 1 < len(non_zero)) and (non_zero[j] == non_zero[j + 1]):
                    merged_value = non_zero[j] * 2
                    merged_row.append(merged_value)
                    score_gain += merged_value
                    j += 2
                else:
                    merged_row.append(non_zero[j])
                    j += 1

            # Preenche a linha no new_board
            for k, v in enumerate(merged_row):
                new_board[i, k] = v

        return new_board, score_gain

    # -------------------------------------------------------
    # Utilitário para IA: codificar estado
    # -------------------------------------------------------
    @staticmethod
    def encode_board(board_4x4: np.ndarray) -> np.ndarray:
        """
        Codifica um tabuleiro 4x4 em um vetor de 16 com log2 dos valores.
        0 (vazio) vira 0.0
        2 -> 1.0, 4 -> 2.0, 8 -> 3.0, ...
        """
        state = []
        for row in board_4x4:
            for val in row:
                if val == 0:
                    state.append(0.0)
                else:
                    state.append(float(np.log2(val)))
        return np.array(state, dtype=np.float32)


# -----------------------------------------------------------
# Heurística simples (professor)
# -----------------------------------------------------------
def evaluate_board(board_4x4: np.ndarray) -> float:
    """
    Heurística simples:
    - mais células vazias é melhor
    - tiles maiores são melhores
    """
    empty_cells = np.count_nonzero(board_4x4 == 0)
    max_tile = board_4x4.max() if board_4x4.size > 0 else 0

    max_component = 0.0
    if max_tile > 0:
        max_component = float(np.log2(max_tile))

    # Peso simples: espaços vazios + 0.1 * log2(max_tile)
    return empty_cells + 0.1 * max_component


def choose_action_heuristic(env: Game2048) -> int:
    """
    Dado um ambiente, testa todas as ações possíveis e escolhe
    aquela que leva ao tabuleiro com melhor avaliação.
    """
    board = env.board
    best_action = None
    best_score = -float('inf')

    for action in range(4):
        new_board, _ = env._move(board, action)
        # Se o movimento não muda nada, ignora (ação inválida naquele estado)
        if np.array_equal(new_board, board):
            continue

        score = evaluate_board(new_board)

        if score > best_score:
            best_score = score
            best_action = action

    # Se nenhuma ação foi válida (jogo travado), devolve 0 por padrão
    if best_action is None:
        best_action = 0

    return best_action
