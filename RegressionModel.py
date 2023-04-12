import pandas as pd
import numpy as np
import torch
import torch.nn
import torch.utils.data
import sys
import matplotlib.pyplot as plt

class SequenceDataSet(torch.utils.data.TensorDataset):
    def __init__(self, X, y, lengths):
		super().__init__(X, y)
		self.lengths = lengths

    def __getitem__(self, index):
		data, target = super().__getitem__(index)
		lengths = self.lengths[index]
		return data, target, lengths 

class EloPredictionNet(torch.nn.Module):
    def __init__(self):
		super().__init__()
        self.hidden_size = 10

        self.lstm = torch.nn.LSTM(1, self.hidden_size)
        
        self.eloScorer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ReLU(true),
            torch.nn.Linear(self.hidden_size, 2)
        )
    
    def forward(self, inputs, lengths):
		nSequence = inputs.size()[0]

        yn = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, enforce_sorted=False)
        yn, _ = self.lstm(yn)
        yn, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(yn)

        # use lens_unpacked
        indices = lens_unpacked-1
		indices = torch.unsqueeze(indices, 1)
		indices = torch.unsqueeze(indices, 2)
		indices = torch.repeat_interleave(indices, self.hiddenSize, dim=2).to(device)
        # yn of last sequence
		yn = torch.gather(yn, 1, indices).squeeze()
		
        yn = self.last(yn)
		return yn.squeeze()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
neginf = -sys.maxsize - 1

# training parameters
N = 1000
validationPercent = 0.05
batchSize = 32


df = pd.read_pickle("KaggleData/dataframe.pickle.zip")

moveTensors = list(map(torch.Tensor, df["MoveScores"][0:N]))
lengths = torch.tensor(list(map(len, moveTensors)))[0:N]
white_elo = torch.tensor(df["white_elo"])[0:N]
black_elo = torch.tensor(df["black_elo"])[0:N]

moveTensors = torch.nn.utils.rnn.pad_sequence(moveTensors)

print(moveTensors.size())


dataset = SequenceDataSet(moveTensors, labels, lengths)
# train, validation split

trainData, validationData = torch.utils.data.random_split(dataset, lengths=[1.0 - validationPercent, validationPercent])

loader = torch.utils.data.DataLoader(trainData, batch_size=batchSize, drop_last=True, num_workers=2)
validationLoader = torch.utils.data.DataLoader(validationData, batch_size=batchSize, drop_last=True,)


