import torch

import chess
import chess.pgn
import pickle

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

def decodeBoard(boardArray):
	# board array is the 1x72 encoding now lets convert it to a board
	board = chess.Board()
	for i in range(64):
		piece = boardArray[0,i]
		maxPiece = max(piece_codes.values())
		minPiece = min(piece_codes.values())
		if maxPiece >= piece >= minPiece:
			square = chess.square_name(i)
			# get the key that matches piece value
			if piece > 0:
				piece = chess.Piece(piece, chess.WHITE)
			else:
				piece = chess.Piece(abs(piece), chess.BLACK)
			board.set_piece_at(i, piece)

	# white or black move
	return board





if __name__ == "__main__":

	games = []
	elos = []
	limit = 50000
	with open("./data.pgn") as file:
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

			if len(games) == limit:
				break
			game = chess.pgn.read_game(file)


	with open("./encodedGames.pickle", "wb") as file:
		pickle.dump(games, file)
		pickle.dump(elos, file)
