import pickle
import pandas as pd
import numpy as np
import torch

df = pd.read_pickle("KaggleData/dataframe.pickle")

boardTensors = list(map(torch.Tensor, df["boards"]))
boardTensors = torch.nn.utils.rnn.pad_sequence(boardTensors, batch_first=True)
print(boardTensors.shape)

moveTensors = list(map(torch.Tensor, df["MoveScores"]))
moveTensors = torch.nn.utils.rnn.pad_sequence(moveTensors, batch_first=True)
print(boardTensors[0,:,:])