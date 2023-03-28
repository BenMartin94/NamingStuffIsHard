import pandas as pd
import numpy as np
import torch
import torch.nn
import torch.utils.data
import sys
import matplotlib.pyplot as plt


# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# for finding non evaluated moves
neginf = -sys.maxsize - 1

# number of examples to train with
N = 500
df = pd.read_pickle("KaggleData/dataframe.pickle.zip")

boardTensors = list(map(torch.Tensor, df["boards"][0:N]))
lengths = torch.tensor(list(map(len, boardTensors)))

maxLength = 100
minLength = 20

boardTensors = torch.nn.utils.rnn.pad_sequence(boardTensors, batch_first=True)
arg = torch.nonzero(torch.logical_and(lengths >= minLength, lengths < maxLength), as_tuple=False)

# remove sequences of annoying lengths
boardTensors = boardTensors[arg, 0:maxLength, :].squeeze()
lengths = lengths[arg].squeeze()

print(boardTensors.size())

# moveTensors = list(map(torch.Tensor, df["MoveScores"][0:N]))
# moveTensors = torch.nn.utils.rnn.pad_sequence(moveTensors, batch_first=True)

# # find missing moves without evaluations, and set to zero padding
# missingMoves = torch.where(moveTensors == neginf)
# moveTensors[missingMoves] = 0
# boardTensors[missingMoves] = 0

Xtrain = boardTensors.float()

white_elo = torch.tensor(df["white_elo"][0:N])[:, None]
black_elo = torch.tensor(df["black_elo"][0:N])[:, None]
ytrain = torch.hstack([white_elo, black_elo]).float()
# ymean = torch.mean(ytrain, 0, keepdim=True)
# ystd = torch.std(ytrain, 0, keepdim=True)
# ytrain = (ytrain - ymean) / ystd
ytrain /= 1000.0
ytrain = ytrain[arg].squeeze()

# parameters for training
lr = 1e-3
nEpochs = 50
batchSize = 32
validationPercent = 0.1

# assumes batch_first=true
class ChessBoardDataset(torch.utils.data.TensorDataset):
	def __init__(self, X, y, lengths):
		super().__init__(X, y)
		self.piecemap = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]
		self.lengths = lengths

	# for when using convolutional layer
	# def __getitem__(self, index):
	# 	data, target = super().__getitem__(index)
	# 	# process batch into chess boards
	# 	shape = data.size()
	# 	# chessBoardTensor = torch.zeros(shape[0], shape[1], 64, 17)
	# 	chessBoardTensor = torch.zeros(shape[0], 17, 64)
		
	# 	# encode each chess piece
	# 	for i in range(len(self.piecemap)):
	# 		chessBoardTensor[...,i,:] = data[...,0:64] == self.piecemap[i]

	# 	# 12->encoding of white move or not
	# 	# 13->white kingside castle
	# 	# 14->white queenside castle
	# 	# 15->black kindside castle
	# 	# 16->black queenside castle
	# 	chessBoardTensor[...,12:17,:] = data[..., 64:69, None]
	# 	# 17, 18, 19 -> en passant square, half move clock, full move number

	# 	# return chessBoardTensor.reshape(shape[0], shape[1], 8, 8, 17), target
	# 	return chessBoardTensor.reshape(shape[0], 17, 8, 8).squeeze(), target.squeeze()

	def __getitem__(self, index):
		data, target = super().__getitem__(index)
		lengths = self.lengths[index]
		return data, target, lengths 
	
dataset = ChessBoardDataset(Xtrain, ytrain, lengths)
trainData, validationData = torch.utils.data.random_split(dataset, lengths=[1.0 - validationPercent, validationPercent])

# dataset = torch.utils.data.TensorDataset(Xtrain, ytrain)
loader = torch.utils.data.DataLoader(trainData, batch_size=batchSize, drop_last=True, num_workers=2)
validationLoader = torch.utils.data.DataLoader(validationData, batch_size=batchSize, drop_last=True, num_workers=2)

class Net(torch.nn.Module):
	def __init__(self, batchSize, device):
		super().__init__()
		# parameters for network
		self.hiddenSize = 100
		self.outputSize = 2

		self.batchSize = batchSize
		self.inputSize = 44

		# self.conv2 = torch.nn.Conv2d(17, nConv, (3, 3))
		self.lstm = torch.nn.LSTM(self.inputSize, self.hiddenSize, batch_first=True)
		self.fclast = torch.nn.Linear(self.hiddenSize, self.outputSize)
		self.fc1 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
		self.relu = torch.nn.ReLU()
		self.conv = torch.nn.Conv2d(1, 1, (3, 3))

		# self.h0 = torch.zeros(1, self.batchSize, self.hiddenSize).to(device)
		# self.c0 = torch.zeros(1, self.batchSize, self.hiddenSize).to(device)
		
	def forward(self, inputs, lengths):
		nSequence = inputs.size()[1]
		# hn = self.h0
		# cn = self.c0
		# for i in range(nSequence):
		# 	# yn = self.conv2(inputs[:,i,...])
		# 	# yn = torch.reshape(yn, (self.batchSize, 1, self.inputSize))
		# 	# yn = self.tanh(yn)
		# 	yn = self.fc1(inputs[:, i,...])
		# 	yn = self.relu(yn)
		# 	yn = self.fc2(yn)
		# 	yn = self.relu(yn)
		# 	yn = torch.reshape(yn, (self.batchSize, 1, self.hiddenSize))
		# 	yn, (hn, cn) = self.lstm(yn, (hn, cn))

		# grab the first 64 elements of the last dim
		justPos = inputs[..., 0:64]
		justPos = torch.reshape(justPos, (self.batchSize*nSequence, 1, 8, 8))
		justPos = self.conv(justPos)
		# add relu
		justPos = self.relu(justPos)

		# now reshape just pos back to batch, sequence, 64
		justPos = torch.reshape(justPos, (self.batchSize, nSequence, 36))
		# now add on the other elements we lost from before
		inputs = torch.cat((justPos, inputs[..., 64:]), dim=2)

		yn = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
		yn, _ = self.lstm(yn)
		yn, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(yn, batch_first=True)
		
		indices = lens_unpacked-1
		indices = torch.unsqueeze(indices, 1)
		indices = torch.unsqueeze(indices, 2)
		indices = torch.repeat_interleave(indices, self.hiddenSize, dim=2).to(device)
		yn = torch.gather(yn, 1, indices).squeeze()

		yn = self.fc1(yn)
		yn = self.relu(yn)
		yn = self.fclast(yn)

		return yn.squeeze()
	

net = Net(batchSize, device).to(device)

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(net.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.9)
trainingHistory = []
validationHistory = []
# optimization loop
for epoch in range(nEpochs):
	print("Beginning Epoch " + str(epoch))
	running_loss = 0.0
	for i, (data, target, lengths) in enumerate(loader):
		data, target = data.to(device), target.to(device)

		optim.zero_grad()

		outputs = net(data, lengths)
		
		loss = criterion(outputs, target)
		loss.backward()

		optim.step()
		running_loss += loss.item()
		
	print(f'[{epoch + 1}] loss: {running_loss:.3f}')
	trainingHistory.append(running_loss)

	validation_loss = 0.0
	with torch.no_grad():
		for i, (data, target, lengths) in enumerate(validationLoader):
			data, target = data.to(device), target.to(device)
			outputs = net(data, lengths)
		
			loss = criterion(outputs, target)
			validation_loss += loss.item()
		print(f'[{epoch + 1}] validation loss: {validation_loss:.3f}')
		validationHistory.append(validation_loss)

# now plot the training history as semilog
plt.semilogy(trainingHistory, label="Training Loss")
plt.semilogy(validationHistory, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.show()

torch.save({
            'epoch': nEpochs,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': running_loss,
            }, 'model.state')

with torch.no_grad():
	for i, (data, target, lengths) in enumerate(loader):
		data, target = data.to(device), target.to(device)
		outputs = net(data, lengths)
		
		for i in range(len(outputs)):
			print(f"[{i}] {1000*outputs[i, :]} {1000*target[i,:]}")
		break
	
