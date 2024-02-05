import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# from game_nn_fast import play
from game_nn_random_move import play
import numba

import pickle as pkl

from game import Game


class AIClient:
    def __init__(self, args, board_size=3, winning_streak=3, random_moves=False):
        self.n = board_size
        self.m = winning_streak
        self.random_moves = random_moves
        if isinstance(args, nn.modules.container.Sequential):
            self.model = args
        else:
            layer_sizes = args
            n_hidden_layers = len(layer_sizes)
            model_sequence = [nn.Linear(self.n ** 2, layer_sizes[0]), nn.ReLU()]
            for i in range(n_hidden_layers - 1):
                model_sequence.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                model_sequence.append(nn.ReLU())
            model_sequence.append(nn.Linear(layer_sizes[-1], 1))
            model_sequence.append(nn.Sigmoid())
            self.model = nn.Sequential(*model_sequence)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            client = pkl.load(f)
        return client

    def save(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self, f)

    def ask_move(self, game):
        empty_slots_indices = np.where(game.board_array == 0)[0]
        nn_outputs = np.zeros(len(empty_slots_indices))
        for i in range(len(empty_slots_indices)):
            input_layer = np.copy(game.board_array)
            input_layer[empty_slots_indices[i]] = game.status
            input_layer = torch.tensor(input_layer, dtype=torch.float32)
            nn_outputs[i] = self.model(input_layer).item()
        if game.status == -1:
            nn_outputs = 1 - nn_outputs
        if self.random_moves:
            if np.sum(nn_outputs) == 0:
                nn_outputs += 1
            probabilities = nn_outputs / np.sum(nn_outputs)
            chosen_move_index = np.random.choice(empty_slots_indices, p=probabilities)
        else:
            chosen_move_index = empty_slots_indices[np.argmax(nn_outputs)]
        return chosen_move_index

    def ai_vs_ai(self, another_ai, n_games):
        n_self_wins = 0
        n_draws = 0
        n_another_wins = 0

        for _ in range(n_games):
            players = [self, another_ai]
            current_player_index = np.random.randint(2)
            game = Game(board_size=self.n, wining_streak=self.m)
            while game.status != 0:
                game.move(players[current_player_index].ask_move(game))
                current_player_index = 1 - current_player_index
            if game.winner == 0:
                n_draws += 1
                continue
            if (1 - current_player_index) == 0:
                n_self_wins += 1
            else:
                n_another_wins += 1

        return n_self_wins, n_draws, n_another_wins

    def train(self, n_steps, batch_size=1, goal_loss=0., print_loss=True):
        loss = None

        def loss_func(output, target, n_before_last_move):
            loss = 3 * torch.mean(((output - target) ** 2) / (1 + n_before_last_move))
            # loss = torch.mean((output - target) ** 2)
            return loss

        optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        for learning_step in range(n_steps):
            n_layers = len(self.model) // 2
            w = []
            b = []
            for i in range(n_layers):
                w.append(self.model[2 * i].weight.clone().detach().numpy().astype(float))
                b.append(self.model[2 * i].bias.clone().detach().numpy().astype(float))
            w = numba.typed.List(w)
            b = numba.typed.List(b)
            x, y, y_predictions, n_before_last = play(w, b, n_games=batch_size, n=self.n, m=self.m)
            x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
            y = torch.tensor(y.reshape(-1, 1), requires_grad=True, dtype=torch.float32)
            n_before_last = torch.tensor(n_before_last.reshape(-1, 1), dtype=torch.float32)
            y_predictions = self.model(x)
            loss = loss_func(y_predictions, y, n_before_last)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if print_loss:
                if (learning_step + 1) % 10 == 0:
                    print(learning_step + 1, loss.item())
                    # print(f'Finished step {learning_step + 1}, latest loss {loss}')
            if loss < goal_loss:
                break
        return loss.item()


if __name__ == '__main__':
    # print('Compiling tictactoe...')
    # play(numba.typed.List([np.zeros(9)]), numba.typed.List([np.zeros(1)]), 1)
    # play(numba.typed.List([np.zeros(25)]), numba.typed.List([np.zeros(1)]), 1, n=5)

    # ai_client = AIClient([32, 24, 16])
    # ai_client = AIClient([50, 50, 50, 50])
    # ai_client = AIClient(np.flip(np.arange(9) + 1).astype(int) * 9)
    # ai_client = AIClient.load('ai_client.pkl')

    # ai_client = AIClient([168, 140, 112, 84, 56], board_size=5, winning_streak=4)

    # from random_client import RandomClient
    # benchmark_client = RandomClient()

    # print('Evaluating initial model...')
    # print(AIClient.ai_vs_ai(ai_client, benchmark_client, 100))

    # print('Training the model...')
    # ai_client.train(1000, batch_size=200)
    # print('First training step finished. Continue training with more refined steps...')
    # ai_client.train(1000, batch_size=1000)

    # print('Evaluating trained model...')
    # print(AIClient.ai_vs_ai(ai_client, benchmark_client, 100))
    # ai_client.save('ai_client.pkl')

    # comparing different architectures
    # architectures = [[8 * (i + 1)] for i in range(16)]
    # architectures = [[32] * (i + 1) for i in range(10)]
    # architectures = [[32, 32, 32], [24, 24, 24], [32, 24, 16, 8], [32, 24, 16]]
    # for architecture in architectures:
    #     print(architecture)
    #     ai_client = AIClient(architecture)
    #     loss = ai_client.train(500, batch_size=100, print_loss=True)
    #     # print(len(architecture), loss)
    #     # print(architecture[0], loss)
    #     # print(loss)
    #     # print()

    # benchmark_client = AIClient([50, 50], random_moves=True)
    # ai_client_best = AIClient.load('ai_client_3_3_final.pkl')
    # ai_client_best = AIClient(ai_client_best.model)
    # ai_client_random = AIClient(ai_client_best.model, random_moves=True)

    benchmark_client = AIClient([100, 100], board_size=5, winning_streak=4, random_moves=True)
    ai_client_best = AIClient.load('ai_client.pkl')
    ai_client_random = AIClient(ai_client_best.model, board_size=5, winning_streak=4, random_moves=True)

    n_games = 1000
    print('Random vs Random')
    print(AIClient.ai_vs_ai(benchmark_client, benchmark_client, n_games))
    print('AI Best vs Random')
    print(AIClient.ai_vs_ai(ai_client_best, benchmark_client, n_games))
    print('AI Random vs Random')
    print(AIClient.ai_vs_ai(ai_client_random, benchmark_client, n_games))
    print('AI Random vs AI Best')
    print(AIClient.ai_vs_ai(ai_client_random, ai_client_best, n_games))
    print('AI Random vs AI Random')
    print(AIClient.ai_vs_ai(ai_client_random, ai_client_random, n_games))
    print('AI Best vs AI Best')
    print(AIClient.ai_vs_ai(ai_client_best, ai_client_best, n_games))
