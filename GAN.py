import pandas as pd
import numpy as np
import torch
import torch.nn
from torch.autograd import Variable
import torch.utils.data
import sys
import matplotlib.pyplot as plt
import chess
import chess.svg
from IPython.core.display import SVG
from IPython.core.display_functions import display
from KaggleData.processKaggleData import decodeBoard
from tqdm.auto import tqdm

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'34

# print the python version


# for finding non evaluated moves
neginf = -sys.maxsize - 1

# number of examples to train with
gamesToGrab = 50000
positionsPerGame = 5

df = pd.read_pickle("KaggleData/dataframe.pickle.zip")

gameTensors = list(map(torch.Tensor, df["boards"][0:gamesToGrab]))
lengths = torch.tensor(list(map(len, gameTensors)))
boardPos = []
for i in range(gamesToGrab):
    # just add a random position from each game
    game = gameTensors[i]
    numMoves = game.size()[0]
    if numMoves < 5:
        continue
    # for j in range(positionsPerGame):
    #     #moveToSel = np.random.randint(0, numMoves, 1)
    #     biggerVal=random.randint(1, (numMoves-1)**2)
    #     moveToSel = int(round((biggerVal)**0.5))
    #     boardPos.append(game[moveToSel, :].squeeze())
    for j in range(numMoves - 1):
        boardPos.append(game[j+1, :].squeeze())

N = len(boardPos)
print("Training on " + str(N) + " Board Positions")

# convert board pos to a tensor now
boardTensors = torch.stack(boardPos)
# zscore the board tensors
mean = torch.mean(boardTensors, dim=0)
std = torch.std(boardTensors, dim=0)
boardTensors = (boardTensors - mean) / std


class Generator(torch.nn.Module):
    def __init__(self, latentSize):
        super().__init__()
        # make a generator now with output being 1x72
        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), padding='same')
        self.relu = torch.nn.ReLU()
        self.BN1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 128, (3, 3), padding='same')
        self.BN2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 1, (3, 3), padding='same')
        self.BN3 = torch.nn.BatchNorm2d(1)
        self.dense = torch.nn.Linear(latentSize, 72)

    def forward(self, x):
        batchSize = x.size()[0]
        noise = torch.reshape(x, (batchSize, 1, 10, 10))
        out = self.conv1(noise)
        out = self.relu(out)
        out = self.BN1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.BN2(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.BN3(out)

        out = torch.reshape(out, (batchSize, x.size()[1]))
        out = self.dense(out)

        return out

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # make a discriminator now with input being 1x72
        self.dense = torch.nn.Linear(72, 1)
        self.conv1 = torch.nn.Conv2d(1, 64, (5, 5), padding='same')
        self.BN1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 1, (5, 5), padding='same')
        self.BN2 = torch.nn.BatchNorm2d(1)

        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        batchSize = x.size()[0]
        pieces = x[..., 0:64]
        extraCrap = x[..., 64:]
        # now reshpae the pieces part into an 8x8 square
        pieces = torch.reshape(pieces, (batchSize, 1, 8, 8))
        # do a conv now
        out = self.conv1(pieces)
        out = self.relu(out)
        out = self.BN1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.BN2(out)
        out = torch.reshape(out, (batchSize, 64))
        out = torch.cat([out, extraCrap], dim=1)
        return self.sig(self.dense(out))



# parameters for training
lr = 1e-4
nEpochs = 100
batchSize = 4096
validationPercent = 0.1
loadModel = False

if loadModel:
    # gen = torch.load("gen.state")
    # disc = torch.load("disc.state")
    # generator = Generator(100).to(device)
    # discriminator = Discriminator().to(device)
    # generator.load_state_dict(gen['model_state_dict'])
    # discriminator.load_state_dict(disc['model_state_dict'])
    generator = torch.load("gen.pth")
    discriminator = torch.load("disc.pth")

else:
    # make the generator and discriminator
    generator = Generator(100).to(device)
    discriminator = Discriminator().to(device)

    # make the optimizers
    genOpt = torch.optim.Adam(generator.parameters(), lr=lr)
    discOpt = torch.optim.Adam(discriminator.parameters(), lr=lr)


    # make the loss function
    loss = torch.nn.BCELoss()
    mseLoss = torch.nn.MSELoss()

    for epoch in tqdm(range(nEpochs), desc="Epochs", colour="purple"):
        # train the discriminator
        for i in tqdm(range(200), desc="Discriminator", colour="green"):
            # sample some data
            idx = np.random.randint(0, N, batchSize)
            data = boardTensors[idx, :].to(device)

            # generate some fake data
            noise = torch.randn(batchSize, 100).to(device)
            fake = generator(noise)

            # train the discriminator
            discOpt.zero_grad()
            realPred = discriminator(data)
            fakePred = discriminator(fake)
            realLoss = loss(realPred, torch.ones(batchSize, 1).to(device))
            fakeLoss = loss(fakePred, torch.zeros(batchSize, 1).to(device))
            discLoss = realLoss + fakeLoss
            discLoss.backward()
            discOpt.step()

        # train the generator
        for i in tqdm(range(200), desc="Generator", colour="green"):
            # generate some fake data
            noise = torch.randn(batchSize, 100).to(device)
            fake = generator(noise)

            # train the generator
            genOpt.zero_grad()
            fakePred = discriminator(fake)
            advLoss = loss(fakePred, torch.ones(batchSize, 1).to(device))
            # lets penalize the generator for any values that came back greater than 6 or less than -6
            outerLoss = torch.where(fake > 6, torch.ones_like(fake), torch.zeros_like(fake)) + torch.where(fake < -6, torch.ones_like(fake), torch.zeros_like(fake))
            # there should never be more than 32 pieces on the board
            empties = torch.where(abs(fake) < 0.5, torch.zeros_like(fake), torch.ones_like(fake))  # puts a 1 in the full spaces
            # for now lets just encourage sparser boards

            outerLoss = outerLoss[:, 0:64]  # only look at the board
            empties = empties[:, 0:64]  # only look at the board
            # now use mse loss for both
            outerLoss = mseLoss(outerLoss, torch.zeros_like(outerLoss))
            emptiesLoss = mseLoss(fake, torch.zeros_like(fake))


            # give them grads
            #outerLoss = torch.tensor(outerLoss, requires_grad=True)
            #emptiesLoss = torch.tensor(emptiesLoss, requires_grad=True)

            genLoss = 0.1 * emptiesLoss + advLoss + outerLoss
            genLoss.backward()
            genOpt.step()

        print("Epoch: %d, Gen Loss: %f, Disc Loss: %f" % (epoch, genLoss, discLoss))


    # torch.save({
    #             'epoch': nEpochs,
    #             'model_state_dict': generator.state_dict(),
    #             'optimizer_state_dict': genOpt.state_dict(),

    #             }, 'gen.state')

    # torch.save({
    #     'epoch': nEpochs,
    #     'model_state_dict': discriminator.state_dict(),
    #     'optimizer_state_dict': discOpt.state_dict(),

    # }, 'disc.state')
    torch.save(generator, "gen.pth")
    torch.save(discriminator, "disc.pth")


# now confirm the discriminator is better than random chance
with torch.no_grad():
    

    # lets take a look at the generator now
    for i in range(10):
        noise = torch.randn(1, 100).to(device)
        givenBoard = generator(noise)
        # waht did the discriminator think of this
        isFake = discriminator(givenBoard)
        print("Should be close to 1")
        print(isFake)
        # now lets decode this board
        givenBoard = givenBoard.cpu()
        givenBoard = givenBoard * std + mean

        givenBoard = givenBoard.numpy()
        # reverse the zscore
        givenBoard = givenBoard.round(0)
        actualBoard = decodeBoard(givenBoard)
        # create a board

        hmm = chess.svg.board(board=actualBoard, size=400)
        # write this to a file
        with open("./GeneratedBoards/board%d.svg" % i, 'w') as f:
            f.write(hmm)
        j=1






