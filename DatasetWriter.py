import pickle
from Utils import *
pgnFile = "./../lichess_db_standard_rated_2013-01.pgn"

# create the pairs list and unique moves list
pairs, uniqueMoves = dataSetGenerator(pgnFile, 1000)
# pickle it
with open('pairs.pickle', 'wb') as handle:
    pickle.dump(pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(uniqueMoves, handle, protocol=pickle.HIGHEST_PROTOCOL)


