import pandas as pd
import numpy as np
import torch
import torch.nn
import torch.utils.data
import sys
import matplotlib.pyplot as plt
import polars
from math import floor
from scipy.special import expit as sigmoid

torch.set_default_dtype(torch.float)


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


class PolarsDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, N) -> None:
        super().__init__()
        self.frame = polars.scan_parquet(filepath)
        self.N = N
        # mini batching in the dataset
        self.batchsize = 1000
        self.offset = 0

    def __getitem__(self, index):
        batchIndex = index % self.batchsize
        # create a new batch if needed
        if batchIndex == 0:
            self.setupNextBatch()
            self.offset += self.batchsize

        return (
            self.boards[batchIndex],
            self.labels[batchIndex],
            self.lengths[batchIndex],
        )

    def __len__(self):
        return self.N

    def setupNextBatch(self):
        frameSlice = self.frame.slice(self.offset, self.batchsize).collect()

        size = frameSlice.height

        # the actual labels
        white_elo = frameSlice["white_elo"]
        black_elo = frameSlice["black_elo"]

        # postions evaluations
        evals = frameSlice["evals"]
        evals_idx = frameSlice["evals_idx"]
        mate_evals = frameSlice["mate_evals"]
        mate_evals_idx = frameSlice["mate_evals_idx"]

        # the positions themselves
        pawns = frameSlice["pawns"]
        bishops = frameSlice["bishops"]
        knights = frameSlice["knights"]
        rooks = frameSlice["rooks"]
        queens = frameSlice["queens"]
        kings = frameSlice["kings"]
        white = frameSlice["white_mask"]
        black = frameSlice["black_mask"]

        # length of each sequence, needed to pack for pytorch
        self.lengths = [pawns[h].shape[0] for h in range(size)]

        # evals -> win probability
        self.winChances = []
        for i in range(size):
            winChance = torch.zeros(self.lengths[i])

            # sigmoid of evals
            winChance[evals_idx[i] - 1] = torch.from_numpy(sigmoid(evals[i].to_numpy()))

            # sign of mate evals
            winChance[mate_evals_idx[i] - 1] = torch.from_numpy(
                np.sign(mate_evals[i].to_numpy())
            ).float()

            self.winChances.append(winChance)

        # evals -> board state
        self.boards = []
        for i in range(size):
            seqlen = self.lengths[i]
            # 12 channels: 6 pieces plus black and white
            rawBoard = torch.zeros(seqlen, 12, 8, 8)

            board = rawBoard.view(seqlen, 12, 64)

            # white pawns
            board[:, 0, :] = torch.from_numpy(
                np.unpackbits(
                    (pawns[i].to_numpy() & white[i].to_numpy()).view(np.uint8)
                ).reshape(seqlen, 64)
            )
            # white bishops
            board[:, 1, :] = torch.from_numpy(
                np.unpackbits(
                    (bishops[i].to_numpy() & white[i].to_numpy()).view(np.uint8)
                ).reshape(seqlen, 64)
            )
            # white knights
            board[:, 2, :] = torch.from_numpy(
                np.unpackbits(
                    (knights[i].to_numpy() & white[i].to_numpy()).view(np.uint8)
                ).reshape(seqlen, 64)
            )
            # white rooks
            board[:, 3, :] = torch.from_numpy(
                np.unpackbits(
                    (rooks[i].to_numpy() & white[i].to_numpy()).view(np.uint8)
                ).reshape(seqlen, 64)
            )
            # white queens
            board[:, 4, :] = torch.from_numpy(
                np.unpackbits(
                    (queens[i].to_numpy() & white[i].to_numpy()).view(np.uint8)
                ).reshape(seqlen, 64)
            )
            # white kings
            board[:, 5, :] = torch.from_numpy(
                np.unpackbits(
                    (kings[i].to_numpy() & white[i].to_numpy()).view(np.uint8)
                ).reshape(seqlen, 64)
            )

            # black pawns
            board[:, 6, :] = torch.from_numpy(
                np.unpackbits(
                    (pawns[i].to_numpy() & black[i].to_numpy()).view(np.uint8)
                ).reshape(seqlen, 64)
            )
            # black bishops
            board[:, 7, :] = torch.from_numpy(
                np.unpackbits(
                    (bishops[i].to_numpy() & black[i].to_numpy()).view(np.uint8)
                ).reshape(seqlen, 64)
            )
            # black knights
            board[:, 8, :] = torch.from_numpy(
                np.unpackbits(
                    (knights[i].to_numpy() & black[i].to_numpy()).view(np.uint8)
                ).reshape(seqlen, 64)
            )
            # black rooks
            board[:, 9, :] = torch.from_numpy(
                np.unpackbits(
                    (rooks[i].to_numpy() & black[i].to_numpy()).view(np.uint8)
                ).reshape(seqlen, 64)
            )
            # black queens
            board[:, 10, :] = torch.from_numpy(
                np.unpackbits(
                    (queens[i].to_numpy() & black[i].to_numpy()).view(np.uint8)
                ).reshape(seqlen, 64)
            )
            # black kings
            board[:, 11, :] = torch.from_numpy(
                np.unpackbits(
                    (kings[i].to_numpy() & black[i].to_numpy()).view(np.uint8)
                ).reshape(seqlen, 64)
            )

            self.boards.append(rawBoard)

        # labels
        white_elo = torch.from_numpy(white_elo.to_numpy())[:, None]
        black_elo = torch.from_numpy(black_elo.to_numpy())[:, None]

        self.labels = torch.cat([white_elo, black_elo], dim=1).float()


N = 2_974_929
data = "/Users/bantingl/Documents/LichessData/BoardInfoFrameLarge.parquet"
dataset = PolarsDataset(data, N)

data, labels, lengths = dataset[0]
print(data.shape, " ", labels.shape, " ", lengths)
exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# df = pd.read_pickle("SavedGameN_10070.pickle.zip")
frames = []
for i in range(1, 19):
    frames.append(pd.read_pickle(f"../LichessData/DataSetChunk_{i:03d}.pickle.zip"))

df = pd.concat(frames, ignore_index=True)
print(df)
# exit()
# parameters for training
N = 19332
P = 50
lr = 1e-4
batchSize = 2048
nEpochs = 100
trainModel = False
loadModel = True

rng = np.random.default_rng()

labels = []
boards = []

nullCount = 0
neginf = -sys.maxsize - 1

for i in range(N):
    evals = df["evals"][i]
    encodings = df["encodings"][i]

    seqlen = len(evals)

    nSamples = min(seqlen // 4, P)
    samples = rng.beta(4, 2, nSamples)
    samples = np.unique(np.floor(samples * (seqlen)).astype(int))

    for s in samples:
        eval = evals[s]

        if eval == neginf:
            nullCount += 1
            continue
        boards.append(encodings[s])
        labels.append(eval)
print(f"Number of null boards detected: {nullCount}")
# exit()

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

print(f"Size of training set: {len(trainSet)}")

## train the model

net = CentipawnNet().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

bestLoss = float("inf")
PATH = "CentipawnModel.chk"
if loadModel:
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    # bestLoss = checkpoint['loss']
    print(f"Loaded model parameters of with loss: {bestLoss} at epoch: {epoch}")

if not trainModel:
    nEpochs = 0

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

    if validationLoss < bestLoss:
        bestLoss = validationLoss
        print(f"Saving model at epoch: {epoch}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": validationLoss,
            },
            PATH,
        )


# test model
net.eval()
true = []
predict = []

# testLoss = torch.nn.MSELoss(reduction='none')

for i, (data, target) in enumerate(validationLoader):
    data, target = data.to(device), target.to(device)

    outputs = net.forward(data)

    true.extend(target.cpu().detach().numpy())
    predict.extend(outputs.cpu().detach().numpy())
    # loss = testLoss(outputs, target)

plt.figure()
plt.scatter(true, predict)
plt.xlabel("true")
plt.ylabel("predict")
plt.title("Centipawn Loss")
plt.savefig("CP_validation.png")
