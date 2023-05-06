import chess
import chess.pgn
import pickle
import pandas as pd
import sys
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

def readGames():
	games = []
	elos = []
	limit = 50000
	with open("KaggleData/data.pgn") as file:
		game = chess.pgn.read_game(file)

		
		while game != None:
			board = chess.Board()
			boards = []

			for move in game.mainline_moves():
				board.push(move)
				vec = encodeBoard(board)
				boards.append(vec)

			games.append(boards)

			if len(games) <= 25000:
				blackElo = int(game.headers["BlackElo"])
				whiteElo = int(game.headers["WhiteElo"])
			elos.append([whiteElo, blackElo])

			if len(games) % 1000 == 0:
				print(f"Processed {len(games)} games")

			if len(games) == limit:
				break
			game = chess.pgn.read_game(file)
	
	elos = np.array(elos)
	return games, elos

def readScores():
	neginf = -sys.maxsize - 1
	df = pd.read_csv("KaggleData/stockfish.csv.zip")
	df["MoveScores"] = [[int(s) if s!= 'NA' else neginf for s in line.split()] for line in df["MoveScores"]]

	return df
		

games, elos = readGames()
df = readScores()
df["boards"] = games
df["white_elo"] = elos[:,0]
df["black_elo"] = elos[:,1]

df.to_pickle("KaggleData/dataframe.pickle.zip")