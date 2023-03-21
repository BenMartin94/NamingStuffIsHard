import pandas as pd
import numpy as np
import torch
import torch.nn
import torch.utils.data
import sys

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for finding non evaluated moves
neginf = -sys.maxsize - 1

# number of examples to train with
N = 25000
df = pd.read_pickle("KaggleData/dataframe.pickle.zip")

boardTensors = list(map(torch.Tensor, df["boards"][0:N]))
boardTensors = torch.nn.utils.rnn.pad_sequence(boardTensors, batch_first=True)

moveTensors = list(map(torch.Tensor, df["MoveScores"][0:N]))
moveTensors = torch.nn.utils.rnn.pad_sequence(moveTensors, batch_first=True)

# find missing moves without evaluations, and set to zero padding
missingMoves = torch.where(moveTensors == neginf)
moveTensors[missingMoves] = 0
boardTensors[missingMoves] = 0

lengths = torch.tensor([t.size()[0] for t in moveTensors])

# Xtrain = torch.nn.utils.rnn.pack_padded_sequence(boardTensors.float(), lengths, batch_first=True, enforce_sorted=False)
Xtrain = boardTensors.float()

white_elo = torch.tensor(df["white_elo"][0:N])[:, None]
black_elo = torch.tensor(df["white_elo"][0:N])[:, None]
ytrain = torch.hstack([white_elo, black_elo]).float()

# parameters for network
inputSize = 72
hiddenSize = 200
outputSize = 2

# parameters for training
lr = 1e-4
nEpochs = 10
batchSize = 256

# dataloader
dataset = torch.utils.data.TensorDataset(Xtrain, ytrain)
loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize)

class Net(torch.nn.Module):
	def __init__(self, inputSize, hiddenSize, outputSize):
		super().__init__()
		self.rnn = torch.nn.LSTM(inputSize, hiddenSize, batch_first=True)
		self.fc1 = torch.nn.Linear(hiddenSize, hiddenSize)
		self.last = torch.nn.Linear(hiddenSize, outputSize)
		self.relu = torch.nn.ReLU()

	def forward(self, inputs):
		yn, hn = self.rnn(inputs)
		yn = self.fc1(yn[:, -1, :])
		yn = self.relu(yn)
		yn = self.last(yn)
		return yn

net = Net(inputSize, hiddenSize, outputSize).to(device)

criterion = torch.nn.MSELoss()
optim = torch.optim.SGD(net.parameters(), lr=lr)

# optimization loop
for epoch in range(nEpochs):
	running_loss = 0.0
	for i, (data, target) in enumerate(loader):
		data, target = data.to(device), target.to(device)

		optim.zero_grad()

		outputs = net(data)
		loss = criterion(outputs, target)
		loss.backward()

		optim.step()
		running_loss += loss.item()
		
		if i % 10 == 9:
			print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')	
		running_loss = 0.0