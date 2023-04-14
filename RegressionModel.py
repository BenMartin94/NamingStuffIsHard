import pandas as pd
import numpy as np
import torch
import torch.nn
import torch.utils.data
import sys
import matplotlib.pyplot as plt

# assumes batch_first=true
class ChessBoardDataset(torch.utils.data.TensorDataset):
	def __init__(self, X, y, lengths):
		super().__init__(X, y)
		self.piecemap = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]
		self.lengths = lengths

	# for when using convolutional layer
	def __getitem__(self, index):
		data, target = super().__getitem__(index)
		# process batch into chess boards
		shape = data.size()
		chessBoardTensor = torch.zeros(shape[0], 18, 64)
		
		# encode each chess piece
		for i in range(len(self.piecemap)):
			chessBoardTensor[...,i,:] = data[...,0:64] == self.piecemap[i]

		# 12->encoding of white move or not
		# 13->white kingside castle
		# 14->white queenside castle
		# 15->black kindside castle
		# 16->black queenside castle
		chessBoardTensor[..., 12:17, :] = data[..., 64:69, None]
        
		#17->movescore
		chessBoardTensor[..., 17, :] = data[..., 72, None]
                
		return chessBoardTensor.reshape(shape[0], 18, 8, 8).squeeze(), target.squeeze(), self.lengths[index]

class SequenceDataset(torch.utils.data.TensorDataset):
    def __init__(self, X, y, lengths):
        super().__init__(X, y)
        self.lengths = lengths
        self.X = X
        self.y = y

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        lengths = self.lengths[index]
        return data, target, lengths

    def __len__(self):
        return len(self.lengths)

class EloPredictionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 384
        self.num_layers = 1

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(18, 64, (3,3), stride=1, dilation=2),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 96, (3,3), stride=1),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(96),
        )

        self.rnn = torch.nn.LSTM(
            input_size=384, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True,
        )

        self.eloScorer = torch.nn.Sequential(
            torch.nn.Linear(2*self.hidden_size,  self.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_size, 2),
        )

    def forward(self, yn, lengths):
        # store original shape
        shape = yn.size()
        
        # flatten batch and sequence for conv2d
        yn = torch.flatten(yn, start_dim=0, end_dim=1)
        yn = self.features(yn)
        
		# unflatten
        yn = torch.unflatten(yn, 0, (shape[0], shape[1]))
        
		# flatten for lstm
        yn = torch.flatten(yn, 2)
        
		# lstm stuff
        yn = torch.nn.utils.rnn.pack_padded_sequence(
            yn, lengths, enforce_sorted=False, batch_first=True
        )
        yn, (hn, cn) = self.rnn(yn)
        hn = hn[self.num_layers-1,:].squeeze()
        # cn = cn[self.num_layers-1,:].squeeze()

        yn, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(yn, batch_first=True)
        indices = lens_unpacked - 1
        indices = torch.unsqueeze(indices, 1)
        indices = torch.unsqueeze(indices, 2)
        indices = torch.repeat_interleave(indices, self.hidden_size, dim=2).to(device)
        yn = torch.gather(yn, 1, indices).squeeze()

        allStates = torch.hstack([yn, hn])
        yn = self.eloScorer(allStates)
        return yn.squeeze()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
neginf = -sys.maxsize - 1

# training parameters
N = 25000
validationPercent = 0.05
batchSize = 1024
lr = 1e-5
nEpochs = 200
trainModel = False

# load in data
df = pd.read_pickle("KaggleData/dataframe.pickle.zip")

moveTensors = list(map(torch.Tensor, df["MoveScores"][0:N]))
boardTensors = list(map(torch.Tensor, df["boards"][0:N]))

lengths = torch.tensor(list(map(len, moveTensors)))[0:N]
white_elo = torch.tensor(df["white_elo"])[0:N]
black_elo = torch.tensor(df["black_elo"])[0:N]
labels = torch.hstack([white_elo[:, None], black_elo[:, None]]).float()


keptGames = torch.nonzero(
    torch.logical_and(lengths >= 10, lengths <= 100)
).squeeze()

labels = labels[keptGames, ...]
lengths = lengths[keptGames].squeeze()

# padded sequences
moveTensors = [moveTensors[i] for i in keptGames]
moveTensors = torch.nn.utils.rnn.pad_sequence(
    moveTensors, batch_first=True
).float()[..., None]

boardTensors = [boardTensors[i] for i in keptGames]
boardTensors = torch.nn.utils.rnn.pad_sequence(
    boardTensors, batch_first=True
).float()

boardsAndMoves = torch.cat([boardTensors, moveTensors],dim=2)


# if missing move evaluation, set to zero
unevaluatedMoves = torch.nonzero(moveTensors == neginf)
moveTensors[unevaluatedMoves] = 0

mean = torch.mean(labels, 0, keepdim=True)
std = torch.std(labels, 0, keepdim=True)
normLabels = (labels - mean) / std

dataset = ChessBoardDataset(boardsAndMoves, normLabels, lengths)

# train, validation split
trainData, validationData = torch.utils.data.random_split(
    dataset, lengths=[1.0 - validationPercent, validationPercent]
)

trainLoader = torch.utils.data.DataLoader(
    trainData, batch_size=batchSize, num_workers=2, shuffle=True
)
validationLoader = torch.utils.data.DataLoader(validationData, batch_size=batchSize)

net = EloPredictionNet().to(device)
criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(net.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, 0.9, verbose=True)

bestLoss = float('inf')
PATH = "./convolutionalNetWeights.chk"

if trainModel:
	for epoch in range(nEpochs):
		epochLoss = 0.0
		net.train()
		for i, (data, target, lengths) in enumerate(trainLoader):
			data, target = data.to(device), target.to(device)

			optim.zero_grad()

			outputs = net.forward(data, lengths)

			loss = criterion(outputs, target)
			loss.backward()

			optim.step()
			epochLoss += loss.item()

		# validate
		validationLoss = 0.0
		
		net.eval()
		for i, (data, target, lengths) in enumerate(validationLoader):
			data, target = data.to(device), target.to(device)

			outputs = net.forward(data, lengths)

			loss = criterion(outputs, target)
			validationLoss += loss.item()

		print(f"[{epoch + 1}] loss: {epochLoss:.3e} validation loss: {validationLoss:.3e}")
		# scheduler.step()

		if validationLoss < bestLoss:
			bestLoss = validationLoss
			
			torch.save({
				'epoch': epoch,
				'model_state_dict': net.state_dict(),
				'optimizer_state_dict': optim.state_dict(),
				'loss': epochLoss,
				}, PATH)
else:
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded model parameters of with loss: {loss} at epoch: {epoch}")
    net.eval()

# test model
black = []
white = []

blackAvg = []
whiteAvg = []

avg = torch.mean(normLabels, dim=0).to(device)

net.eval()
testLoss = torch.nn.MSELoss(reduction='none')
with torch.no_grad():
    for i, (data, target, lengths) in enumerate(validationLoader):
        data, target = data.to(device), target.to(device)

        outputs = net.forward(data, lengths)

        for j in range(10):
            print(f"Label: {target[j,:]} prediction: {outputs[j, :]}")

        errors = testLoss(outputs, target)
        black.extend(errors[:,1])
        white.extend(errors[:,0])
        
        zeros = torch.zeros_like(target).to(device)
        avgErrors = testLoss(zeros, target)

        blackAvg.extend(avgErrors[:,1])
        whiteAvg.extend(avgErrors[:,0])


# create tolerance curves
white = torch.tensor(white)
black = torch.tensor(black)

whiteAvg = torch.tensor(whiteAvg)
blackAvg = torch.tensor(blackAvg)

whiteMax = torch.max(white)
blackMax = torch.max(black)

whiteAvgMax = torch.max(whiteAvg)
blackAvgMax = torch.max(blackAvg)

whiteMax = torch.max(whiteMax, whiteAvgMax)
blackMax = torch.max(blackMax, blackAvgMax)

white /= whiteMax
black /= blackMax

whiteAvg /= whiteMax
blackAvg /= blackMax

steps = torch.linspace(0, 1.0, 100)
total = white.size()[0]
whiteCounts = [torch.count_nonzero(white <= tol) / total for tol in steps]
blackCounts = [torch.count_nonzero(black <= tol) / total for tol in steps]

whiteAvgCounts = [torch.count_nonzero(whiteAvg <= tol) / total for tol in steps]
blackAvgCounts = [torch.count_nonzero(blackAvg <= tol) / total for tol in steps]

plt.figure()
plt.plot(steps, whiteCounts, '-')
plt.plot(steps, whiteAvgCounts, '-.')
plt.xlabel("Normalized Error")
plt.ylabel("Percentage of Examples Less than Normalized Error")
plt.title("White Elo")
plt.savefig("white.png")

plt.figure()
plt.plot(steps, blackCounts, '-')
plt.plot(steps, blackAvgCounts, '-.')
plt.xlabel("Normalized Error")
plt.ylabel("Percentage of Examples Less than Normalized Error")
plt.title("Black Elo")
plt.savefig("black.png")
