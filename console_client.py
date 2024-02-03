class ConsoleClient:
    def __init__(self):
        pass

    def ask_move(self, game):
        print(game)

        while True:
            print('Enter your move (example: a1)')
            coordinates = input().lower()
            try:
                col = ord(coordinates[0]) - ord('a')
                row = int(coordinates[1:]) - 1
                if (row < 0) or (row >= game.n) or (col < 0) or (col >= game.n):
                    print('Illegal move')
                    continue
                if game.board_array[game.n * row + col] != 0:
                    print('Illegal move')
                    continue
                return game.n * row + col
            except ValueError as e:
                print(e)
