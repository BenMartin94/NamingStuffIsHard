import pandas as pd
import numpy as np
import torch
import torch.nn
import torch.utils.data
import sys

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# for finding non evaluated moves
neginf = -sys.maxsize - 1

# number of examples to train with
N = 5000
df = pd.read_pickle("KaggleData/dataframe.pickle.zip")

boardTensors = list(map(torch.Tensor, df["boards"][0:N]))
boardTensors = torch.nn.utils.rnn.pad_sequence(boardTensors, batch_first=True)

moveTensors = list(map(torch.Tensor, df["MoveScores"][0:N]))
moveTensors = torch.nn.utils.rnn.pad_sequence(moveTensors, batch_first=True)

# find missing moves without evaluations, and set to zero padding
missingMoves = torch.where(moveTensors == neginf)
moveTensors[missingMoves] = 0
boardTensors[missingMoves] = 0

Xtrain = boardTensors.float()

white_elo = torch.tensor(df["white_elo"][0:N])[:, None]
black_elo = torch.tensor(df["white_elo"][0:N])[:, None]
ytrain = torch.hstack([white_elo, black_elo]).float()

# parameters for network
inputSize = 72
hiddenSize = 50
outputSize = 2

# parameters for training
lr = 1e-2
nEpochs = 2
batchSize = 128

# dataloader
dataset = torch.utils.data.TensorDataset(Xtrain, ytrain)
loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize)

class Net(torch.nn.Module):
	def __init__(self, inputSize, hiddenSize, outputSize):
		super().__init__()
		self.rnn = torch.nn.LSTM(inputSize, hiddenSize, hiddenSize, batch_first=True)
		self.fc = torch.nn.Linear(hiddenSize, outputSize)

	def forward(self, inputs):
		yn, hn = self.rnn(inputs)
		yn = self.fc(yn[:,-1,:])
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
		
		print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
		running_loss = 0.0