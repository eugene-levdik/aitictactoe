import numpy as np


class RandomClient:
    def __init__(self):
        pass

    def ask_move(self, game):
        possible_moves = game.possible_moves()
        return np.random.choice(possible_moves)
