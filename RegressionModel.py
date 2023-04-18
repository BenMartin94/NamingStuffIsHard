import pandas as pd
import numpy as np
import torch
import torch.nn
import torch.utils.data
import sys
import matplotlib.pyplot as plt
from PolarsData import PolarsDataset, PolarsDataStream

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
        self.hidden_size = 256
        
        self.features1 = torch.nn.Sequential(
            torch.nn.Conv2d(18, 96, (4,4)), 
            torch.nn.ReLU(), # 6 x 6
            torch.nn.Conv2d(96, 256, (4,4)),
            torch.nn.ReLU(), # 4x4
            torch.nn.Conv2d(256, 64, (1,1)),
            torch.nn.ReLU(), # 4x4
        )
        
        self.rnn = torch.nn.LSTM(
            input_size=self.hidden_size, 
            hidden_size=self.hidden_size, 
            batch_first=True,
        )
        
        self.eloScorer = torch.nn.Sequential(
            torch.nn.Linear(2*self.hidden_size,  self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, 2),
        )

    def forward(self, yn, lengths):
        # store original shape
        shape = yn.size()
        batch = shape[0]
        seq = shape[1]
        
        # flatten batch and sequence for conv2d
        yn = torch.flatten(yn, start_dim=0, end_dim=1)
        yn = yn.view(batch*seq, -1, 8, 8)
        yn = self.features1(yn)
        
        # unflatten
        yn = yn.view(batch, seq, -1)
		
		# lstm stuff
        yn = torch.nn.utils.rnn.pack_padded_sequence(
            yn, lengths, enforce_sorted=False, batch_first=True
        )
        yn, (hn, cn) = self.rnn(yn)
        
        yn, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(yn, batch_first=True)
        
        hn = hn.squeeze()
        
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
validationPercent = 0.05
batchSize = 1024
lr = 1e-3
trainModel = True
loadModel = False
saveModel = False

# N = 2_974_929
# N = 18_387
N = 198_285
data = "/Users/bantingl/Documents/LichessData/BoardInfoFrameMedLarge.parquet"
# dataset = PolarsDataset(data, N, batch_size=batchSize)
dataset = PolarsDataStream(data, N)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False)
# dataset.normalizationParams()


net = EloPredictionNet().to(device)
criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(net.parameters(), lr=lr)

bestLoss = float('inf')
PATH = 'ConvolutionalEloModel.state'

if loadModel:
	checkpoint = torch.load(PATH)
	net.load_state_dict(checkpoint['model_state_dict'])
	optim.load_state_dict(checkpoint['optimizer_state_dict'])
	
epochLoss = 0.0
if trainModel:
	net.train()
	for i, (data, target, lengths) in enumerate(dataloader):
		data, target = data.to(device), target.to(device)

		# reduce size of predictions
		target = (target - 1530) / 370
		
		optim.zero_grad()

		outputs = net.forward(data, lengths)

		loss = criterion(outputs, target)
		loss.backward()

		optim.step()
		epochLoss += loss.item()

		print(f"nBatches: {i} batch loss: {epochLoss} ")
		epochLoss = 0.0
		
		if saveModel:
			torch.save({
				'model_state_dict': net.state_dict(),
				'optimizer_state_dict': optim.state_dict(),
				}, PATH)
exit()
# validate model
net.eval()
testLoss = torch.nn.MSELoss(reduction='none')

black = []
white = []
blackTrue = []
whiteTrue = []

errorsWhite = []
whiteNorms = []

with torch.no_grad():
    batch = dataset[0]
    
    data = batch[0]
    target = batch[1]
    lengths = batch[2]
    
    data, target = data.to(device), target.to(device)

	# reduce size of predictions
    target = (target - 1530) / 370
    
    outputs = net.forward(data, lengths)
    
    blackTrue.extend(target[:,1].cpu().numpy())
    whiteTrue.extend(target[:,0].cpu().numpy())
    
    black.extend(outputs[:,1].cpu().numpy())
    white.extend(outputs[:,0].cpu().numpy())
    
	# see errors
    exampleLoss = testLoss(outputs, target)
    
    errorsWhite = exampleLoss[:,0].cpu().numpy()
    whiteNorms = (target[:,0]**2).cpu().numpy()
    
    for j in range(10):
        print(f"label {target[j,:]} prediction: {outputs[j,:]}")     


steps = np.linspace(0, np.max(whiteNorms), 100)
total = whiteNorms.shape[0]
whiteCounts = [np.count_nonzero(errorsWhite <= tol) / total for tol in steps]
# blackCounts = [np.count_nonzero(black <= tol) / total for tol in steps]

whiteAvgCounts = [np.count_nonzero(whiteNorms <= tol) / total for tol in steps]
# blackAvgCounts = [np.count_nonzero(blackAvg <= tol) / total for tol in steps]

plt.figure()
plt.plot(steps, whiteCounts, '-')
plt.plot(steps, whiteAvgCounts, '-.')
plt.xlabel("Unnormalized Error")
plt.ylabel("Percentage of Examples Less than Normalized Error")
plt.title("White Elo Prediction Tolerance Curve")
plt.savefig("white_tolerance.png")


plt.figure()
plt.scatter(whiteTrue, white)
plt.title("White Elo")
plt.xlabel("True")
plt.ylabel("Prediction")
plt.savefig("white_elo_scatter.png")

plt.figure()
plt.scatter(blackTrue, black)
plt.title("Black Elo")
plt.xlabel("True")
plt.ylabel("Prediction")
plt.savefig("black_elo_scatter.png")
exit()    

# test model
black = []
white = []

blackAvg = []
whiteAvg = []

avg = torch.mean(normLabels, dim=0).to(device)

testWhite = []
predictWhite = []

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
        
        testWhite.extend(target[:,0].cpu())
        predictWhite.extend(outputs[:,0].cpu())

plt.figure()
plt.scatter(testWhite, predictWhite)
plt.plot([-2, 2], [-2,2], 'k--')
plt.title("White Elo")
plt.xlabel("True")
plt.ylabel("Prediction")
plt.savefig("scatter.png")

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
