import numpy as np
from numba import njit


@njit
def rand_choice_nb(arr, prob):
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@njit
def calculate_target_values(board_evaluations, game_result):
    n = len(board_evaluations)
    # return game_result * np.ones(n)

    weighted_y = np.zeros(n)
    weighted_y[:-1] = board_evaluations[1:]
    weighted_y[-1] = game_result
    # return weighted_y

    # weighted_y *= np.flip(np.arange(n) + 1)
    weighted_y *= np.arange(n) + 1
    average_sequence = np.zeros(n)
    weight_sum = 0
    sequence_sum = 0
    for i in range(n):
        # weight_sum += 1
        # weight_sum += i + 1
        weight_sum += n - i + 1
        sequence_sum += weighted_y[-i - 1]
        average_sequence[-i - 1] = sequence_sum / weight_sum
    return average_sequence


@njit
def check_game_over(board):
    board2d = board.reshape(int(np.sqrt(board.size)), -1)
    row_sum = np.sum(board2d, axis=0)
    col_sum = np.sum(board2d, axis=1)
    diagonal_sum = board2d[0, 0] + board2d[1, 1] + board2d[2, 2]
    anti_diagonal_sum = board2d[0, 2] + board2d[1, 1] + board2d[2, 0]
    if np.max(row_sum) == 3 or np.max(col_sum) == 3 or diagonal_sum == 3 or anti_diagonal_sum == 3:
        return 1  # X wins
    if np.min(row_sum) == -3 or np.min(col_sum) == -3 or diagonal_sum == -3 or anti_diagonal_sum == -3:
        return -1  # O wins
    if 0 not in board:
        return 0  # Draw
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
def play(w, b, n_games, n=3):
    x = np.zeros(((n ** 2 + 1) * n_games, n ** 2))
    y = np.zeros((n ** 2 + 1) * n_games)
    y_prediction = np.zeros((n ** 2 + 1) * n_games)
    n_before_last = np.zeros((n ** 2 + 1) * n_games)
    next_board_ind = 0
    for i_game in range(n_games):
        winner = None
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
            if current_player == 1:
                probabilities = nn_outputs
            else:
                probabilities = 1 - nn_outputs
            probabilities = probabilities / np.sum(probabilities)
            chosen_move_index = rand_choice_nb(possible_moves, probabilities)
            board[chosen_move_index] = current_player

            y_prediction[next_board_ind] = nn_outputs[np.where(possible_moves == chosen_move_index)[0][0]]
            x[next_board_ind] = board
            next_board_ind += 1

            winner = check_game_over(board)
            if winner is not None:
                break

            current_player = -current_player
        target = (winner + 1) / 2
        y[game_start_ind:next_board_ind] = calculate_target_values(y_prediction[game_start_ind:next_board_ind], target)
        n_before_last[game_start_ind:next_board_ind] = np.flip(np.arange(next_board_ind - game_start_ind))
    return x[:next_board_ind], y[:next_board_ind], y_prediction[:next_board_ind], n_before_last[:next_board_ind]


if __name__ == '__main__':
    import torch
    import numba.typed.typedlist

    from ai_client import AIClient
    model = AIClient([10, 10, 10]).model
    x = torch.zeros(9)
    x[0] = 1
    print(model(x).item())

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
