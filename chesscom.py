import requests
import chess.pgn
import chess.engine
import io
import asyncio


async def main():
	r = requests.get("https://api.chess.com/pub/player/magnuscarlsen/games/2023/03")
	gameString = r.json()['games'][0]['pgn']

	pgn = chess.pgn.read_game(io.StringIO(gameString))
	board = pgn.board()


	transport, engine = await chess.engine.popen_uci("E:/Projects/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe")


	for move in pgn.mainline_moves():
		info = await engine.analyse(board, chess.engine.Limit(depth=18))
		print(info["score"])

		board.push(move)

	await engine.quit()
		
	

asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
asyncio.run(main())
