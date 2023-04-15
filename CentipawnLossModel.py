import pandas as pd
import numpy as np
import torch
import torch.nn
import torch.utils.data
import sys
import matplotlib.pyplot as plt


class ChessBoardDataset(torch.utils.data.TensorDataset):
    def __init__(self, X, y):
        super().__init__(X, y)
        self.piecemap = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]

    # for when using convolutional layer
    def __getitem__(self, index):
        data, target = super().__getitem__(index)

        chessBoardTensor = torch.zeros(17, 64)

        # encode each chess piece
        for i in range(len(self.piecemap)):
            chessBoardTensor[i, :] = data[0:64] == self.piecemap[i]

        # 12->encoding of white move or not
        # 13->white kingside castle
        # 14->white queenside castle
        # 15->black kindside castle
        # 16->black queenside castle
        chessBoardTensor[12:17, :] = data[64:69, None]

        return chessBoardTensor.reshape(17, 8, 8).squeeze(), target.squeeze()


class CentipawnNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 50

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(17, 32, (5, 5)),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, (3, 3)),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(64),
        )

        self.boardScorer = torch.nn.Sequential(
            torch.nn.Linear(64 * 2 * 2, self.hidden_size),
            torch.nn.Tanh(),
            torch.nn.BatchNorm1d(self.hidden_size),
            torch.nn.Linear(self.hidden_size, 1),
        )

    def forward(self, x):
        batch = x.shape[0]
        x = self.conv(x)
        x = x.view(batch, -1)
        x = self.boardScorer(x)
        return x.squeeze()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_pickle("SavedGameN_10070.pickle.zip")

# parameters for training
N = 5000
P = 20
lr = 1e-3
batchSize = 2048
nEpochs = 10

rng = np.random.default_rng()

labels = []
boards = []

for i in range(N):
    evals = df["evals"][i]
    encodings = df["encodings"][i]

    seqlen = len(evals)

    nSamples = min(seqlen // 4, P)
    samples = rng.beta(4, 2, nSamples)
    samples = np.floor(samples * seqlen).astype(int)

    for s in samples:
        boards.append(encodings[s])
        eval = evals[s]
        # eval = min(eval, 250)
        # eval = max(eval, -250)
        labels.append(eval)

torch.set_default_dtype(torch.float)

X = torch.tensor(boards).float()
y = torch.tensor(labels).float()

mean = torch.mean(y)
std = torch.std(y)
y = (y - mean) / std

dataset = ChessBoardDataset(X, y)

trainSet, validationSet = torch.utils.data.random_split(dataset, [0.95, 0.05])
trainLoader = torch.utils.data.DataLoader(
    trainSet, batch_size=batchSize, num_workers=2, shuffle=True
)
validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=batchSize)


## train the model

net = CentipawnNet().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

for epoch in range(nEpochs):
    epochLoss = 0.0
    net.train()
    for i, (data, target) in enumerate(trainLoader):
        data, target = data.to(device), target.to(device)

        optim.zero_grad()

        outputs = net.forward(data)
        loss = criterion(outputs, target)
        loss.backward()

        optim.step()
        epochLoss += loss.item()

    # validate
    validationLoss = 0.0

    net.eval()
    for i, (data, target) in enumerate(validationLoader):
        data, target = data.to(device), target.to(device)

        outputs = net.forward(data)

        loss = criterion(outputs, target)
        validationLoss += loss.item()

    print(f"[{epoch + 1}] loss: {epochLoss:.3e} validation loss: {validationLoss:.3e}")
    # scheduler.step()
