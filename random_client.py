import numpy as np


class RandomClient:
    def __init__(self):
        pass

    def ask_move(self, game):
        empty_slots_indices = np.where(game.board_array == 0)[0]
        return np.random.choice(empty_slots_indices)
