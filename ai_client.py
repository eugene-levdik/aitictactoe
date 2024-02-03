import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from game_nn_fast import play
import numba

import pickle as pkl

from game import Game


class AIClient:
    def __init__(self, args, n=3):
        self.n = n
        if isinstance(args, nn.modules.container.Sequential):
            self.model = args
        else:
            layer_sizes = args
            n_hidden_layers = len(layer_sizes)
            model_sequence = [nn.Linear(n ** 2, layer_sizes[0]), nn.ReLU()]
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
        # probabilities = nn_outputs / np.sum(nn_outputs)
        # chosen_move_index = np.random.choice(empty_slots_indices, p=probabilities)
        chosen_move_index = empty_slots_indices[np.argmax(nn_outputs)]
        return chosen_move_index

    def ai_vs_ai(self, another_ai, n_games):
        n_self_wins = 0
        n_draws = 0
        n_another_wins = 0

        for _ in range(n_games):
            players = [self, another_ai]
            current_player_index = np.random.randint(2)
            game = Game()
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

    def train(self, n_steps, batch_size=1, goal_loss=0.):
        def loss_func(output, target, n_before_last_move):
            loss = torch.mean(((output - target) ** 2) / (1 + n_before_last_move))
            # loss = torch.mean((output - target) ** 2)
            return loss

        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        for learning_step in range(n_steps):
            n_layers = len(self.model) // 2
            w = []
            b = []
            for i in range(n_layers):
                w.append(self.model[2 * i].weight.clone().detach().numpy().astype(float))
                b.append(self.model[2 * i].bias.clone().detach().numpy().astype(float))
            w = numba.typed.List(w)
            b = numba.typed.List(b)
            x, y, y_predictions, n_before_last = play(w, b, n_games=batch_size)
            x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
            y = torch.tensor(y.reshape(-1, 1), requires_grad=True, dtype=torch.float32)
            n_before_last = torch.tensor(n_before_last.reshape(-1, 1), dtype=torch.float32)
            y_predictions = self.model(x)
            loss = loss_func(y_predictions, y, n_before_last)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (learning_step + 1) % 10 == 0:
                print(f'Finished step {learning_step + 1}, latest loss {loss}')
            if loss < goal_loss:
                return


if __name__ == '__main__':
    ai_client = AIClient([30, 20, 10])
    # ai_client = AIClient([50, 50, 50, 50])
    # ai_client = AIClient(np.flip(np.arange(9) + 1).astype(int) * 9)
    # ai_client = AIClient.load('ai_client.pkl')

    from random_client import RandomClient
    benchmark_client = RandomClient()

    print('Compiling tictactoe...')
    play(numba.typed.List([np.zeros(9)]), numba.typed.List([np.zeros(1)]), 1)

    print('Evaluating initial model...')
    print(AIClient.ai_vs_ai(ai_client, benchmark_client, 1000))

    print('Training the model...')
    ai_client.train(2000, batch_size=100)
    print('First training step finished. Continue training with more refined steps...')
    ai_client.train(1000, batch_size=1000)

    print('Evaluating trained model...')
    print(AIClient.ai_vs_ai(ai_client, benchmark_client, 1000))
    ai_client.save('ai_client.pkl')
