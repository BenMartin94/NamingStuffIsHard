import requests
import chess.pgn
import io

r = requests.get("https://api.chess.com/pub/player/magnuscarlsen/games/2023/03")
gameString = r.json()['games'][0]['pgn']

pgn = chess.pgn.read_game(io.StringIO(gameString))
board = pgn.board()

for move in pgn.mainline_moves():
	board.push(move)
	print(board, end='\n\n')


