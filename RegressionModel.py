import pandas as pd
import numpy as np
import torch
import torch.nn
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib.patches
from KaggleData import loadKaggleData, loadKaggleTestSet
import pickle

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
            chessBoardTensor[..., i, :] = data[..., 0:64] == self.piecemap[i]

        # 12->encoding of white move or not
        # 13->white kingside castle
        # 14->white queenside castle
        # 15->black kindside castle
        # 16->black queenside castle
        chessBoardTensor[..., 12:17, :] = data[..., 64:69, None]

        # 17->movescore
        chessBoardTensor[..., 17, :] = data[..., 72, None]

        return (
            chessBoardTensor.reshape(shape[0], 18, 8, 8).squeeze(),
            target.squeeze(),
            self.lengths[index],
        )

class EloPredictionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 512
        
        self.features_deep = torch.nn.Sequential(
            torch.nn.Conv2d(18, 64, (3, 3)),
            torch.nn.ReLU(),  # 6 x 6
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(64, 128, (3, 3)),
            torch.nn.ReLU(),  # 4 x 4
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(128, 256, (3, 3)),
            torch.nn.ReLU(),  # 2 x 2
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(256, 512, (2, 2)),
            torch.nn.ReLU(),  # 1 x 1
            torch.nn.Dropout2d(0.2)
        )

        self.rnn = torch.nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )

        self.eloScorer = torch.nn.Sequential(
            torch.nn.Linear(2 * self.hidden_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout1d(0.2),
            torch.nn.Linear(self.hidden_size, 2),
        )

    def forward(self, yn, lengths):
        # store original shape
        shape = yn.size()
        batch = shape[0]
        seq = shape[1]
        
        # flatten batch and sequence for conv2d
        yn = torch.flatten(yn, start_dim=0, end_dim=1)
        yn = yn.view(batch * seq, -1, 8, 8)
        yn = self.features_deep(yn)
        
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

# training parameters
validationPercent = 0.05
batchSize = 384
epochs = 100
lr = 1e-4
trainModel = False
loadModel = True
saveModel = True
testModel = False
testExample = False

SAVE_PATH = "Checkpoints/ConvolutionalEloModel_Deep_L1_Dropout_Kaggle.state"
LOAD_PATH = "RegressionModelWeightsFinal.state"

if testExample:
	net = EloPredictionNet().to(device)
	checkpoint = torch.load(LOAD_PATH)
	net.load_state_dict(checkpoint["model_state_dict"])
	net.eval()
	games = []
	lengths = []
	labels = []
	files = [
        "Games/Martin_vs_Martin.pgn.encoded",
        "Games/Lucas_vs_Martin.pgn.encoded",
        "Games/BenGame.pgn.encoded",
        "Games/Ian_Game1.pgn.encoded",
        "Games/Ian_Game2.pgn.encoded",
        "Games/Ian_Game3.pgn.encoded"
	]
	for f in files:
		with open(f, "rb") as file:
			game = pickle.load(file)
			game = torch.tensor(game)
			games.append(game)
			lengths.append(game.shape[0])
			labels.append([0,0])
	X = torch.nn.utils.rnn.pad_sequence(games, batch_first=True)
	dataset = ChessBoardDataset(X, torch.tensor(labels), lengths)
	data = []
	lens = []
	for i in range(6):
		d, _, l = dataset[i]
		data.append(d.unsqueeze(0))
		lens.append(l)
	data = torch.vstack(data).to(device)
	predict = net.forward(data, lengths)
        
	mean = torch.tensor([2246.8511, 2241.8914]).to(device)
	std = torch.tensor([268.3849, 270.9836]).to(device)
	print(predict*std + mean)
	exit()
        
if testModel:
	Xtest, lengthsTest, targetMean, targetStd = loadKaggleTestSet("KaggleData/dataframe.pickle.zip")
	ytest = torch.zeros((25000, 2)).to(device)
	testSet = ChessBoardDataset(Xtest, ytest, lengthsTest)
	testLoader = torch.utils.data.DataLoader(
		testSet, batch_size=batchSize
	)
	print(targetMean)
	print(targetStd)
	targetMean = targetMean.to(device)
	targetStd = targetStd.to(device)
	
	net = EloPredictionNet().to(device)
	checkpoint = torch.load(LOAD_PATH)
	net.load_state_dict(checkpoint["model_state_dict"])
	net.eval()
	with torch.no_grad():
		events = []
		white_predict = []
		black_predict = []
		id = 25001
		for data, _, lengths in testLoader:
			data = data.to(device)
            
			predict = net.forward(data, lengths)
			predict = predict * targetStd + targetMean
			predict = predict.long().cpu().numpy()
			          
			for i in range(predict.shape[0]):
				events.append(id)
				white_predict.append(predict[i, 0])
				black_predict.append(predict[i, 1])
                                
				id +=1
		df = pd.DataFrame({
                    "Event" : events,
                    "WhiteElo" : white_predict,
                    "BlackElo" : black_predict
                    })
		df.to_csv("KaggleData/Predictions.csv", index=False)
	exit()


# load kaggle data
X, y, lengths = loadKaggleData("KaggleData/dataframe.pickle.zip")
fullset = ChessBoardDataset(X, y, lengths)
dataset, validationset = torch.utils.data.random_split(fullset, [0.95, 0.05])
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batchSize, num_workers=4, persistent_workers=True, shuffle=True
)
validationloader = torch.utils.data.DataLoader(
    validationset, batch_size=batchSize,
)
targetMean = torch.mean(y, dim=0).unsqueeze(0).to(device)
targetStd = torch.std(y, dim=0).unsqueeze(0).to(device)
print(f"Train set\nmean: {targetMean}\nstd: {targetStd}\nsize: {len(fullset)}\ntrain size: {len(dataset)}")

net = EloPredictionNet().to(device)
criterion = torch.nn.L1Loss()
optim = torch.optim.Adam(net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
	optim, 
	verbose=True,
	factor=0.5)
bestLoss = float("inf")

if loadModel:
    checkpoint = torch.load(LOAD_PATH)
    net.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    bestLoss = checkpoint["best_loss"]
    
epochLoss = 0.0
i = 0
alpha = 0.1
movingAverage = 0.0

if trainModel:
	net.train()
	for outer in range(epochs):
		for data, target, lengths in dataloader:
			data, target = data.to(device), target.to(device)
			
			# reduce size of predictions
			target = (target - targetMean) / targetStd
                        
			optim.zero_grad()

			outputs = net.forward(data, lengths)

			loss = criterion(outputs, target)
			loss.backward()

			optim.step()
			epochLoss += loss.item()

			movingAverage = alpha*epochLoss + (1-alpha)*movingAverage
			print(f"[{outer:2d}, {i:4d}] moving average: {movingAverage:.3e} batch loss: {epochLoss:.3e} ")
			epochLoss = 0.0
			i += 1
			
		# validate
		net.eval()
		validationLoss = 0.0
		for data, target, lengths in validationloader:
			data, target = data.to(device), target.to(device)
			target = (target - targetMean) / targetStd
		
			outputs = net.forward(data, lengths)
			validationLoss += criterion(outputs, target).item()
		print(f"VALIDATING at epoch: {i} test loss: {validationLoss}")

		scheduler.step(validationLoss)

		
		if validationLoss < bestLoss:
			bestLoss = validationLoss
			if saveModel:
				torch.save(
					{
						"model_state_dict": net.state_dict(),
						"optimizer_state_dict": optim.state_dict(),
						"scheduler_state_dict" : scheduler.state_dict(),
						"best_loss" : bestLoss
					},
					SAVE_PATH)
		if saveModel:
			torch.save(
				{
					"model_state_dict": net.state_dict(),
					"optimizer_state_dict": optim.state_dict(),
					"scheduler_state_dict" : scheduler.state_dict(),
					"best_loss" : bestLoss
				},
				SAVE_PATH + f"{outer:02d}")
		net.train()
        
# validate model
net.eval()
testLoss = torch.nn.L1Loss(reduction="none")

black = []
white = []
blackTrue = []
whiteTrue = []

errorsWhite = []
whiteNorms = []

errorsBlack = []
blackNorms = []
with torch.no_grad():
    for data, target, lengths in validationloader:
        data, target = data.to(device), target.to(device)
        break
    # reduce size of predictions
    target = (target - targetMean) / targetStd
    test = torch.zeros_like(target).to(device)
    
    outputs = net.forward(data, lengths)

    blackTrue.extend(target[:, 1].cpu().numpy())
    whiteTrue.extend(target[:, 0].cpu().numpy())

    black.extend(outputs[:, 1].cpu().numpy())
    white.extend(outputs[:, 0].cpu().numpy())

    # see errors
    exampleLoss = testLoss(outputs, target)
    
    errorsWhite = exampleLoss[:, 0].cpu().numpy()
    errorsBlack = exampleLoss[:, 1].cpu().numpy()
    
    meanCheck = testLoss(test, target).cpu().numpy()
    whiteNorms = meanCheck[:,0]

    blackNorms = meanCheck[:,1]

    for j in range(10):
        print(f"label {target[j,:]} prediction: {outputs[j,:]}")
        

steps = np.linspace(0, 1, 100)
total = whiteNorms.shape[0]
whiteMax = np.max(whiteNorms)
blackMax = np.max(blackNorms)

whiteCounts = [np.count_nonzero(errorsWhite <= tol*whiteMax) / total for tol in steps]
blackCounts = [np.count_nonzero(errorsBlack <= tol*blackMax) / total for tol in steps]

whiteAvgCounts = [np.count_nonzero(whiteNorms <= tol*whiteMax) / total for tol in steps]
blackAvgCounts = [np.count_nonzero(blackNorms <= tol*blackMax) / total for tol in steps]

plt.figure()
plt.plot(steps, whiteCounts, "-", label="prediction")
plt.plot(steps, whiteAvgCounts, "-.", label="error from using mean")
plt.legend()
plt.xlabel("Normalized Error")
plt.ylabel("Percentage of Examples Less than Normalized Error")
plt.title("White Elo Prediction Tolerance Curve")
plt.savefig("Images/white_tolerance.png")

plt.figure()
plt.plot(steps, blackCounts, "-", label="prediction")
plt.plot(steps, blackAvgCounts, "-.", label="error from using mean")
plt.xlabel("Normalized Error")
plt.legend()
plt.ylabel("Percentage of Examples Less than Normalized Error")
plt.title("Black Elo Prediction Tolerance Curve")
plt.savefig("Images/black_tolerance.png")

fig, ax = plt.subplots()
rect = matplotlib.patches.Rectangle((0,0), 2.5, 1.5, color='tab:green', alpha=0.5)
ax.add_patch(rect)

rect = matplotlib.patches.Rectangle((0,0), -3.5, -1.5, color='tab:green', alpha=0.5)
ax.add_patch(rect)

rect = matplotlib.patches.Rectangle((0,0), -3.5, 1.5, color='tab:red', alpha=0.5)
ax.add_patch(rect)

rect = matplotlib.patches.Rectangle((0,0), 2.5, -1.5, color='tab:red', alpha=0.5)
ax.add_patch(rect)

ax.scatter(whiteTrue, white)
ax.set_title("White Elo")
ax.set_xlabel("True")
ax.set_ylabel("Prediction")
plt.savefig("Images/white_elo_scatter.png")

fig, ax = plt.subplots()
rect = matplotlib.patches.Rectangle((0,0), 2.5, 1.5, color='tab:green', alpha=0.5)
ax.add_patch(rect)

rect = matplotlib.patches.Rectangle((0,0), -3.5, -1.5, color='tab:green', alpha=0.5)
ax.add_patch(rect)

rect = matplotlib.patches.Rectangle((0,0), -3.5, 1.5, color='tab:red', alpha=0.5)
ax.add_patch(rect)

rect = matplotlib.patches.Rectangle((0,0), 2.5, -1.5, color='tab:red', alpha=0.5)
ax.add_patch(rect)

ax.scatter(blackTrue, black)
ax.set_title("Black Elo")
ax.set_xlabel("True")
ax.set_ylabel("Prediction")
plt.savefig("Images/black_elo_scatter.png")
