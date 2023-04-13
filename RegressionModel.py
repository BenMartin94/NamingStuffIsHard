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
        self.hidden_size = 50

        self.rnn = torch.nn.LSTM(
            input_size=1, 
            hidden_size=self.hidden_size, 
            num_layers=1, 
            batch_first=True,
        )

        self.eloScorer = torch.nn.Sequential(
            torch.nn.Linear(3*self.hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )

    def forward(self, yn, lengths):
        # lstm stuff
        yn = torch.nn.utils.rnn.pack_padded_sequence(
            yn, lengths, enforce_sorted=False, batch_first=True
        )
        yn, (hn, cn) = self.rnn(yn)
        yn, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(yn, batch_first=True)
        hn = hn.squeeze()
        cn = cn.squeeze()

        indices = lens_unpacked - 1
        indices = torch.unsqueeze(indices, 1)
        indices = torch.unsqueeze(indices, 2)
        indices = torch.repeat_interleave(indices, self.hidden_size, dim=2).to(device)
        # yn of last sequence
        yn = torch.gather(yn, 1, indices).squeeze()

        allStates = torch.hstack([yn, hn, cn])
        yn = self.eloScorer(allStates)
        return yn.squeeze()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
neginf = -sys.maxsize - 1

# training parameters
N = 25000
validationPercent = 0.05
batchSize = 2048
lr = 1e-3
nEpochs = 100

# load in data
df = pd.read_pickle("KaggleData/dataframe.pickle.zip")

moveTensors = list(map(torch.Tensor, df["MoveScores"][0:N]))
lengths = torch.tensor(list(map(len, moveTensors)))[0:N]
white_elo = torch.tensor(df["white_elo"])[0:N]
black_elo = torch.tensor(df["black_elo"])[0:N]
labels = torch.hstack([white_elo[:, None], black_elo[:, None]]).float()


keptGames = torch.nonzero(
    torch.logical_and(lengths > 10, lengths <= 40)
).squeeze()

labels = labels[keptGames, ...]
lengths = lengths[keptGames].squeeze()

# pytorch dataset
moveTensors = [moveTensors[i] for i in keptGames]
moveTensors = torch.nn.utils.rnn.pad_sequence(
    moveTensors, batch_first=True
).float()[..., None]

# if missing move evaluation, set to zero
unevaluatedMoves = torch.nonzero(moveTensors == neginf)
moveTensors[unevaluatedMoves] = 0

mean = torch.mean(labels, 0, keepdim=True)
std = torch.std(labels, 0, keepdim=True)
normLabels = (labels - mean) / std

dataset = SequenceDataSet(moveTensors, normLabels, lengths)

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
# scheduler = torch.optim.lr_scheduler.StepLR(optim, 50, 0.9, verbose=True)

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

net.eval()
with torch.no_grad():
    for i, (data, target, lengths) in enumerate(validationLoader):
        data, target = data.to(device), target.to(device)

        outputs = net.forward(data, lengths)

        for j in range(10):
            print(f"Label: {target[j,:]} prediction: {outputs[j, :]}")
