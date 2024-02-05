import numpy as np
from numba import njit


@njit
def check_win_on_segment(segment, m):
    segment_length = len(segment)
    segment_sum = np.sum(segment[:m])
    if np.abs(segment_sum) == m:
        return int(np.sign(segment_sum))
    for shift in range(segment_length - m):
        segment_sum += segment[shift + m] - segment[shift]
        if np.abs(segment_sum) == m:
            return int(np.sign(segment_sum))
    return None


@njit
def check_game_over(board, n=3, m=3):
    board2d = board.reshape(n, n)
    # check rows
    for row_index in range(n):
        winner = check_win_on_segment(board2d[row_index, :], m)
        if winner is not None:
            return winner

    # check columns
    for col_index in range(n):
        winner = check_win_on_segment(board2d[:, col_index], m)
        if winner is not None:
            return winner

    # check canonical diagonals
    max_diag_index = n - m
    for diag_index in range(-max_diag_index, max_diag_index + 1):
        winner = check_win_on_segment(np.diag(board2d, diag_index), m)
        if winner is not None:
            return winner

    # check anti-canonical diagonals
    flipped_board = np.fliplr(board2d)
    for diag_index in range(-max_diag_index, max_diag_index + 1):
        winner = check_win_on_segment(np.diag(flipped_board, diag_index), m)
        if winner is not None:
            return winner

    # if there are no more moves available, it's a draw
    is_full = (np.count_nonzero(board) == len(board))
    if is_full:
        return 0

    return None


@njit
def nn_forward(w, b, input_layer):
    # allows multiple layers in the form of [n_neurons x n_layers] matrix
    n_boards = input_layer.shape[1]
    current_layer = input_layer.copy()
    for i in range(len(w) - 1):
        current_b = b[i].repeat(n_boards).reshape((-1, n_boards))
        current_layer = np.dot(w[i], current_layer) + current_b
        current_layer = current_layer * (current_layer > 0)  # ReLU
    last_b = b[-1].repeat(n_boards).reshape((-1, n_boards))
    nn_outputs = np.dot(w[-1], current_layer) + last_b
    nn_outputs = 1 / (1 + np.exp(-nn_outputs))  # Sigmoid
    return nn_outputs[0]


@njit
def play(w, b, n_games, n=3, m=3):
    x = np.zeros(((n ** 2 + 1) * n_games, n ** 2))
    y = np.zeros((n ** 2 + 1) * n_games)
    y_prediction = np.zeros((n ** 2 + 1) * n_games)
    n_before_last = np.zeros((n ** 2 + 1) * n_games)
    next_board_ind = 0
    for i_game in range(n_games):
        game_start_ind = next_board_ind
        board = np.zeros(n ** 2)
        current_player = 1
        for i_move in range(n ** 2):
            possible_moves = np.where(board == 0)[0]
            n_possible_moves = len(possible_moves)
            possible_next_boards = board.repeat(n_possible_moves).reshape((-1, n_possible_moves)).T
            for i_possible_move in range(n_possible_moves):
                possible_next_boards[i_possible_move, possible_moves[i_possible_move]] = current_player
            possible_next_boards = possible_next_boards.T

            nn_outputs = nn_forward(w, b, possible_next_boards)
            if not game_start_ind == next_board_ind:
                y[next_board_ind - 1] = np.abs(np.max(current_player * nn_outputs))
            chosen_move_index = np.random.randint(n_possible_moves)
            y_prediction[next_board_ind] = nn_outputs[chosen_move_index]

            board[possible_moves[chosen_move_index]] = current_player

            x[next_board_ind] = board
            next_board_ind += 1

            winner = check_game_over(board, n=n, m=m)
            if winner is not None:
                game_outcome = (winner + 1) / 2
                y[next_board_ind - 1] = game_outcome
                n_before_last[game_start_ind:next_board_ind] = np.flip(np.arange(next_board_ind - game_start_ind))
                break

            current_player = -current_player

    return x[:next_board_ind], y[:next_board_ind], y_prediction[:next_board_ind], n_before_last[:next_board_ind]


if __name__ == '__main__':
    import torch
    import numba.typed.typedlist

    from ai_client import AIClient

    model = AIClient([10, 10, 10]).model
    x = torch.zeros(9)
    x[0] = 1
    # print(model(x).item())

    n = len(model) // 2
    w = []
    b = []
    for i in range(n):
        w.append(model[2 * i].weight.detach().numpy().astype(float))
        b.append(model[2 * i].bias.detach().numpy().astype(float))
    w = numba.typed.List(w)
    b = numba.typed.List(b)

    x, y, y_predictions, n_before_last = play(w, b, 2)
    print(x)
    print(y)
    print(y_predictions)
    print(n_before_last)
