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

    def _check_win_on_segment(self, segment):
        segment_length = len(segment)
        segment_sum = np.sum(segment[:self.m])
        if np.abs(segment_sum) == self.m:
            return int(np.sign(segment_sum))
        for shift in range(segment_length - self.m):
            segment_sum += segment[shift + self.m] - segment[shift]
            if np.abs(segment_sum) == self.m:
                return int(np.sign(segment_sum))
        return None

    def _check_win(self):
        # check rows
        for row_index in range(self.n):
            winner = self._check_win_on_segment(self.board[row_index, :])
            if winner is not None:
                return winner

        # check columns
        for col_index in range(self.n):
            winner = self._check_win_on_segment(self.board[:, col_index])
            if winner is not None:
                return winner

        # check canonical diagonals
        max_diag_index = self.n - self.m
        for diag_index in range(-max_diag_index, max_diag_index + 1):
            winner = self._check_win_on_segment(np.diag(self.board, diag_index))
            if winner is not None:
                return winner

        # check anti-canonical diagonals
        flipped_board = np.fliplr(self.board)
        for diag_index in range(-max_diag_index, max_diag_index + 1):
            winner = self._check_win_on_segment(np.diag(flipped_board, diag_index))
            if winner is not None:
                return winner

        # if there are no more moves available, it's a draw
        is_full = (np.count_nonzero(self.board_array) == len(self.board_array))
        if is_full:
            return 0

        return None

    def possible_moves(self):
        return np.where(self.board_array == 0)[0]

    def move(self, index):
        if self.status == 0:
            raise ValueError('The game is over')
        if self.board_array[index] != 0:
            raise ValueError('This place is not empty')
        self.board_array[index] = self.status

        winner = self._check_win()
        if winner is not None:
            self.winner = winner
            self.status = 0
            return

        self.status = -self.status

    def move_by_coord(self, coord):
        col = ord(coord[0]) - ord('a')
        row = int(coord[1:]) - 1
        self.move(self.n * row + col)
