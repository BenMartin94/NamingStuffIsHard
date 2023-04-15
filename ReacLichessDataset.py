import zstandard as zstd
import chess
import chess.pgn
import sys
import io
import pandas as pd
import numpy as np

piece_codes = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6, 'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}

def encodeBoard(board):
	encoding = []
	# 64 squares
	for i in range(64):
		piece = board.piece_at(i)
		if piece is not None:
			code = piece_codes[piece.symbol()]
		else:
			code = 0
		encoding.append(code)
	
	# white or black move
	moveCode = -1
	if board.turn == chess.WHITE:
		moveCode = 1
	encoding.append(moveCode)

	# can white castle 0 or 1
	whiteKingCastle = 0
	if board.has_kingside_castling_rights(chess.WHITE):
		whiteKingCastle = 1
	whiteQueenCastle = 0
	if board.has_queenside_castling_rights(chess.WHITE):
		whiteQueenCastle = 1
	encoding.append(whiteKingCastle)
	encoding.append(whiteQueenCastle)

	# can black castle 0 or 1
	blackKingCastle = 0
	if board.has_kingside_castling_rights(chess.BLACK):
		blackKingCastle = 1
	blackQueenCastle = 0
	if board.has_queenside_castling_rights(chess.BLACK):
		blackQueenCastle = 1

	encoding.append(blackKingCastle)
	encoding.append(blackQueenCastle)
	
	# en passant target square
	ep_square = board.ep_square
	if ep_square == None:
		ep_square = -1 # not sure what to input when not possible
	encoding.append(ep_square)

	# half move clock
	encoding.append(board.halfmove_clock)

	# fullmove number
	encoding.append(board.fullmove_number)
	return encoding


PATH = "/Users/bantingl/Documents/LichessData/lichess_db_standard_rated_2023-03.pgn.zst"

N = 10000
totalGames = 0

with open(PATH, "rb") as input:
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(input)
    out = sys.stdout

    savedGames = []

    while totalGames < N:
        print(
            f"----------------------Reading new chunk at N={N}-------------------------"
        )
        chunk = reader.read(2**24)

        stringChunk = chunk.decode("utf-8")
        pgn = io.StringIO(stringChunk)

        while True:
            gameValid = True
            game = chess.pgn.read_game(pgn)

            if game == None:
                break

            if "Event" not in game.headers:
                break
            if "WhiteElo" not in game.headers:
                break
            if "BlackElo" not in game.headers:
                break

            evals = []
            encodings = []
            for move in game.mainline():
                if move.eval() == None:
                    gameValid = False
                    break
                score = move.eval().white().score()
                if score == None:
                    score = 0
                evals.append(int(score))
                
                encoding = encodeBoard(move.board())
                encodings.append(encoding)
            if gameValid:
                if game.headers["Event"] == "Rated Rapid game":
                    savedGames.append(
                        [
                            int(game.headers["WhiteElo"]),
                            int(game.headers["BlackElo"]),
                            evals, 
                            encodings,
						]
                    )
                    totalGames += 1


frame = pd.DataFrame(savedGames, columns=["white", "black", "evals", "encodings"])
print(frame)

elo = frame[["white", "black"]].to_numpy()

mean = np.mean(elo, 0)
std = np.std(elo, 0)

print(f"mean {mean}")
print(f"std {std}")

frame.to_pickle(f"SavedGameN_{totalGames}.pickle.zip")
