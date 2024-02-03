import numpy as np


class Game:

    def __init__(self, board_size=3, wining_streak=3):
        self.n = board_size
        self.m = wining_streak
        self.board_array = np.zeros(self.n ** 2, dtype=int)
        self.status = 1  # 1 -- x move, -1 -- o move, 0 -- end
        self.winner = None  # 1 -- x won, -1 -- o won, 0 -- draw, None -- ongoing game

    @property
    def board(self):
        return self.board_array.reshape(self.n, self.n)

    def __repr__(self):
        repr_str = 'Game status: '
        if self.status == 0:
            if self.winner == 0:
                repr_str += "DRAW"
            else:
                repr_str += f'{"X" if self.winner == 1 else "O"} WON'
        else:
            repr_str += f'{"X" if self.status == 1 else "O"} move'
        repr_str += '\n'

        offset = len(str(self.n))
        repr_str += ' ' * offset + ' ' + ' '.join(list('abcdefghijklmnopqrstuvwxyz'[:self.n])) + ' \n'
        for i in range(self.n):
            repr_str += ' ' * (offset - len(str(i))) + str(i + 1) + '|'
            repr_str += ' '.join(list(map(str, self.board[i, :]))).\
                replace('0', '·').replace('-1', 'o').replace('1', 'x')
            repr_str += '|\n'
        return repr_str

    def _check_win(self):
        # TODO: works only for n-long line winning conditions
        row_max_sum = np.max(abs(np.sum(self.board, axis=0)))
        if self.n == row_max_sum:
            return True

        col_max_sum = np.max(abs(np.sum(self.board, axis=1)))
        if self.n == col_max_sum:
            return True

        main_diag_sum = abs(np.trace(self.board))
        if self.n == main_diag_sum:
            return True

        anti_diag_sum = abs(np.trace(np.fliplr(self.board)))
        if self.n == anti_diag_sum:
            return True

        return None

    def move(self, index):
        if self.status == 0:
            raise ValueError('The game is over')
        if self.board_array[index] != 0:
            raise ValueError('This place is not empty')
        self.board_array[index] = self.status

        if self._check_win():
            self.winner = self.status
            self.status = 0
            return

        # check draw
        if np.count_nonzero(self.board_array) == len(self.board_array):
            self.winner = 0
            self.status = 0
            return

        self.status = -self.status

    def move_by_coord(self, coord):
        col = ord(coord[0]) - ord('a')
        row = int(coord[1:]) - 1
        self.move(self.n * row + col)
