import pickle
import pandas as pd
import numpy as np
import sys

neginf = -sys.maxsize - 1

df = pd.read_csv("KaggleData/stockfish.csv.zip")
df["MoveScores"] = [[int(s) if s!= 'NA' else neginf for s in line.split()] for line in df["MoveScores"]]

with open("KaggleData/encodedGames.pickle", "rb") as file:
	games = pickle.load(file)
	elos = pickle.load(file)
	elos = np.array(elos)

	# unlabelled data in set
	elos[25000:, :] = 0

	df["boards"] = games
	df["white_elo"] = elos[:,0]
	df["black_elo"] = elos[:,1]
	
	df.to_pickle("KaggleData/dataframe.pickle")