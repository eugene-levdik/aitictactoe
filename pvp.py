from game import Game
from console_client import ConsoleClient
from ai_client import AIClient
from random_client import RandomClient


def play_game(x_client, o_client, board_size=3, winning_streak=3):
    game = Game(board_size=board_size, wining_streak=winning_streak)

    while game.status != 0:
        game.move(x_client.ask_move(game))
        if game.status == 0:
            break
        game.move(o_client.ask_move(game))
    return game.winner


if __name__ == '__main__':
    ai_client = AIClient.load('ai_client.pkl')
    # winner = play_game(ai_client, ConsoleClient())
    # winner = play_game(ConsoleClient(), ai_client)
    # winner = play_game(ConsoleClient(), ConsoleClient(), board_size=5, winning_streak=3)
    # winner = play_game(ConsoleClient(), ai_client, board_size=5, winning_streak=4)
    winner = play_game(ai_client, ConsoleClient(), board_size=5, winning_streak=4)
    print('Game is over')
    if winner == 0:
        print('DRAW')
    else:
        print(f'{"X" if winner == 1 else "O"} WON')
