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
black_elo = torch.tensor(df["black_elo"][0:N])[:, None]
ytrain = torch.hstack([white_elo, black_elo]).float()
ymean = torch.mean(ytrain, 0, keepdim=True)
ystd = torch.std(ytrain, 0, keepdim=True)
ytrain = (ytrain - ymean) / ystd

# parameters for network
nConv = 4
hiddenSize = 150
outputSize = 2

# parameters for training
lr = 1e-3
nEpochs = 10
batchSize = 16

# assumes batch_first=true
class ChessBoardDataset(torch.utils.data.TensorDataset):
	def __init__(self, X, y):
		super().__init__(X, y)
		self.piecemap = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]

	def __getitem__(self, index):
		data, target = super().__getitem__(index)
		# process batch into chess boards
		shape = data.size()
		# chessBoardTensor = torch.zeros(shape[0], shape[1], 64, 17)
		chessBoardTensor = torch.zeros(shape[0], 17, 64)
		
		# encode each chess piece
		for i in range(len(self.piecemap)):
			chessBoardTensor[...,i,:] = data[...,0:64] == self.piecemap[i]

		# 12->encoding of white move or not
		# 13->white kingside castle
		# 14->white queenside castle
		# 15->black kindside castle
		# 16->black queenside castle
		chessBoardTensor[...,12:17,:] = data[..., 64:69, None]
		# 17, 18, 19 -> en passant square, half move clock, full move number

		# return chessBoardTensor.reshape(shape[0], shape[1], 8, 8, 17), target
		return chessBoardTensor.reshape(shape[0], 17, 8, 8).squeeze(), target.squeeze()
	
dataset = ChessBoardDataset(Xtrain, ytrain)
loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, drop_last=True, num_workers=2)

class Net(torch.nn.Module):
	def __init__(self, nConv, hiddenSize, outputSize, batchSize, device):
		super().__init__()
		self.batchSize = batchSize
		self.inputSize = nConv * 6 * 6

		self.conv2 = torch.nn.Conv2d(17, nConv, (3, 3))
		self.lstm = torch.nn.LSTM(self.inputSize, hiddenSize, batch_first=True)
		self.last = torch.nn.Linear(hiddenSize, outputSize)
		self.relu = torch.nn.ReLU()
		self.h0 = torch.zeros(1, batchSize, hiddenSize).to(device)
		self.c0 = torch.zeros(1, batchSize, hiddenSize).to(device)
		
	def forward(self, inputs):
		nSequence = inputs.size()[1]
		hn = self.h0
		cn = self.c0
		for i in range(nSequence):
			yn = self.conv2(inputs[:,i,...])
			yn = torch.reshape(yn, (self.batchSize, 1, self.inputSize))
			yn = self.relu(yn)
			yn, (hn, cn) = self.lstm(yn, (hn, cn))
		yn = self.last(yn)
		return yn.squeeze()
	

net = Net(nConv, hiddenSize, outputSize, batchSize, device).to(device)

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(net.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.9)

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
			for label in range(4):
				print(f"predict {outputs[label,:]} label {target[label, :]}")

		running_loss = 0.0
	# scheduler.step()
	
	
