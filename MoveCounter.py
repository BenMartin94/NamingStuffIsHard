import chess.engine
import chess.pgn
import numpy as np

mypgnFile = "./../lichess_db_standard_rated_2013-01.pgn"

def get_games(pgnFile):
    with open(pgnFile) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            yield game



def get_moves(game, uniqueMoves):
    # get all unique moves in a game
    board = game.board()
    print(board)
    for move in game.mainline_moves():
        for candmove in board.legal_moves:
            uniqueMoves.add(candmove)
        board.push(move)
    #print(len(uniqueMoves))
uniqueMoves = set()
numberOfGames = 1000
for game in get_games(mypgnFile):
    get_moves(game, uniqueMoves)
    numberOfGames -= 1
    if numberOfGames == 0:
        break
    print(numberOfGames)
print(len(uniqueMoves))


