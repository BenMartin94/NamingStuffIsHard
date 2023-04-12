import pandas as pd
import numpy as np
import torch
import torch.nn
import torch.utils.data
import sys
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
neginf = -sys.maxsize - 1

N = 1000

df = pd.read_pickle("KaggleData/dataframe.pickle.zip")

moveTensors = list(map(torch.Tensor, df["MoveScores"][0:N]))
lengths = torch.tensor(list(map(len, moveTensors)))[0:N]
white_elo = torch.tensor(df["white_elo"])[0:N]
black_elo = torch.tensor(df["black_elo"])[0:N]




moveTensors = torch.nn.utils.rnn.pad_sequence(moveTensors)

print(moveTensors.size())


