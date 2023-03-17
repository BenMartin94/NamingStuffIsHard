import chess.pgn
import numpy as np

def encode_board(board):
    boardMatrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(i * 8 + j)
            if piece is not None:
                boardMatrix[i][j] = piece_values[piece.symbol()]
    return boardMatrix

class PositionMovePair:
    def __init__(self, position, move, moveNumber):
        self.position = encode_board(position)
        self.move = move
        # index of the unique move
        self.moveNumber = moveNumber
def get_games(pgnFile):
    with open(pgnFile) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            yield game

piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 10, 'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': -10}



def encode_moves(games):
    uniqueMoves = set()
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            for candmove in board.legal_moves:
                uniqueMoves.add(candmove)
            board.push(move)
    return list(uniqueMoves)

def dataSetGenerator(pgnFile, maxGames=1000):
    games = get_games(pgnFile)
    gamesList = []
    for game in games:
        maxGames -= 1
        if maxGames == 0:
            break
        gamesList.append(game)

    uniqueMoves = encode_moves(gamesList)
    # now need to loop through the mainline of each game and create position, move pairs
    # for each move in the game, create a position, move pair
    pairs = []
    couunter = 0
    for game in gamesList:
        print("Generating game number: ", couunter)
        board = game.board()
        for move in game.mainline_moves():
            # find the index into the unique moves list
            moveNumber = uniqueMoves.index(move)

            pairs.append(PositionMovePair(board, move, moveNumber))
            board.push(move)
        couunter += 1
    return pairs, uniqueMoves


