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
import random

import cairosvg
import imageio.v2 as imageio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding
class DDPM(torch.nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, vectorLength=72):
        super(DDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.vectorLength = vectorLength
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta):
        n, l = x0.shape
        abar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n,l).to(self.device)

        noisy = abar.sqrt().reshape(n, 1) * x0 + (1 - abar).sqrt().reshape(n, 1) * eta

        return noisy

    def noisePred(self, x, t):
        # x is noisy
        # t is the time along the diffusion process
        return self.network(x, t)

    def createFromNoise(self, n, t, writeToGif=False, gifName='test.gif'):
        # n is the number of things to create from pure noise
        # x0 is the initial state
        # t is the time along the diffusion process
        svglists = {}
        for i in range(n):
            svglists[i] = []
        with torch.no_grad():
            xN = torch.randn(n, self.vectorLength).to(self.device)
            for idx, t in enumerate(list(range(self.n_steps))[::-1]):
                times = (torch.ones(n, 1) * t).to(self.device).long()
                noisePred = self.noisePred(xN, times)
                alphaT = self.alphas[t]
                alphaTBar = self.alpha_bars[t]

                xN = (1 / alphaT.sqrt()) * (xN - (1 - alphaT) / (1 - alphaTBar).sqrt() * noisePred)

                if writeToGif:
                    for i in range(n):
                        temp = xN[i:i+1,:]
                        # reverse z score
                        temp = temp*std + mean
                        temp = torch.round(temp, decimals=0)
                        board = decodeBoard(temp)
                        svg = chess.svg.board(board=board, size=400)
                        svglists[i].append(svg)

                if t>0:
                    z = torch.randn(n, self.vectorLength).to(self.device)
                    betaT = self.betas[t]
                    sigmaT = betaT.sqrt()
                    xN = xN + sigmaT * z

            if writeToGif:
                for i in range(n):
                    frames = []
                    for svg in svglists[i]:
                        png = cairosvg.svg2png(bytestring=svg)
                        frames.append(imageio.imread(png))
                    imageio.mimsave("./GeneratedBoards/board" + str(i) + ".gif", frames, fps=10)
            return xN

    def createFromReal(self, x0, t):
        # x0 is the initial state
        # t is the time along the diffusion process to reach
        T = t
        with torch.no_grad():
            xN = self.forward(x0, t, None)
            # now weve noised it up, so lets come back from noise
            for idx, t in enumerate(list(range(T))[::-1]):
                times = (torch.ones(1, 1) * t).to(self.device).long()
                noisePred = self.noisePred(xN, times)
                alphaT = self.alphas[t]
                alphaTBar = self.alpha_bars[t]

                xN = (1 / alphaT.sqrt()) * (xN - (1 - alphaT) / (1 - alphaTBar).sqrt() * noisePred)
                if t > 0:
                    z = torch.randn(1, self.vectorLength).to(self.device)
                    betaT = self.betas[t]
                    sigmaT = betaT.sqrt()
                    xN = xN + sigmaT * z
            return xN



class UNet(torch.nn.Module):
    def __init__(self, n_steps=500, time_emb_dim=100):
        super(UNet, self).__init__()
        self.actv = torch.nn.SiLU()

        # Sinusoidal embedding
        self.time_embed = torch.nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)
        self.inConv = torch.nn.Conv2d(1, 16, 7, padding='same')
        # down layer blocks
        self.te1 = self._make_te(time_emb_dim, 1)
        self.down1 = self._make_conv_down(16, 32)
        self.te2 = self._make_te(time_emb_dim, 32)
        self.down2 = self._make_conv_down(32, 64)
        # bottleneck
        self.te3 = self._make_te(time_emb_dim, 64)
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, padding=1),
            self.actv,
            torch.nn.Conv2d(64, 64, 3, padding=1)
        )
        # up layer blocks
        self.te4 = self._make_te(time_emb_dim, 64)
        self.up1 = self._make_conv_up(64, 32)
        self.lateral1 = self._make_conv_only(64, 32)
        self.te5 = self._make_te(time_emb_dim, 32)
        self.up2 = self._make_conv_up(32, 1)

        self.dense = torch.nn.Linear(72, 72)

    def forward(self, x, t):
        # x is batches x 72 so
        batch_size = x.size()[0]
        pieces = x[:, 0:64].reshape(batch_size, 1, 8, 8)
        extra = x[:, 64:72]
        t = self.time_embed(t)
        out0 = self.inConv(pieces)
        out0 = self.actv(out0)
        out1 = self.down1(out0 + self.te1(t).reshape(batch_size, -1, 1, 1))
        out2 = self.down2(out1 + self.te2(t).reshape(batch_size, -1, 1, 1))
        out3 = self.bottleneck(out2 + self.te3(t).reshape(batch_size, -1, 1, 1))

        # now skip conns
        up1 = self.up1(out3 + self.te4(t).reshape(batch_size, -1, 1, 1))
        out4 = torch.cat((out1, up1), dim=1)
        out4 = self.lateral1(out4)
        up2 = self.up2(out4 + self.te5(t).reshape(batch_size, -1, 1, 1))
        # now flatted back to 64
        up2 = up2.reshape(batch_size, 64)
        #up2 = self.actv(up2)
        up2 = torch.cat((up2, extra), dim=1)
        up2 = self.dense(up2)
        return up2

    def _make_te(self, dim_in, dim_out):
        return torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out),
            self.actv,
            torch.nn.Linear(dim_out, dim_out)
        )

    def _make_conv_down(self, dim_in, dim_out):
        return torch.nn.Sequential(
            torch.nn.Conv2d(dim_in, dim_out, 3, padding=1),
            self.actv,
            torch.nn.Conv2d(dim_out, dim_out, 3, padding=1),
            torch.nn.MaxPool2d(2)
        )

    def _make_conv_up(self, dim_in, dim_out):
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(dim_in, dim_out, 4, stride=2, padding=1),
            self.actv,

        )

    def _make_conv_only(self, dim_in, dim_out):
        return torch.nn.Sequential(
            torch.nn.Conv2d(dim_in, dim_out, 3, padding=1),
            self.actv,
        )



if __name__ == "__main__":
    # for finding non evaluated moves
    neginf = -sys.maxsize - 1
    epochs = 50
    loadModel = False
    modelName = "LateGame.pth"
    batch_size = 64
    # number of examples to train with
    gamesToGrab = 5000
    positionsPerGame = 2


    df = pd.read_pickle("KaggleData/dataframe.pickle.zip")

    gameTensors = list(map(torch.Tensor, df["boards"][0:gamesToGrab]))
    lengths = torch.tensor(list(map(len, gameTensors)))
    boardPos = []
    for i in range(gamesToGrab):
        # just add a random position from each game
        game = gameTensors[i]
        numMoves = game.size()[0]
        for j in range(positionsPerGame):
            #moveToSel = np.random.randint(0, numMoves, 1)
            biggerVal=random.randint(0, (numMoves-1)**4)
            moveToSel = int(round((biggerVal)**0.25))
            boardPos.append(game[moveToSel, :].squeeze())

    N = len(boardPos)
    # convert board pos to a tensor now
    boardTensors = torch.stack(boardPos)
    trainBoardTensors = boardTensors[0:N-10]
    # zscore it
    mean = trainBoardTensors.mean(dim=0)
    std = trainBoardTensors.std(dim=0)
    trainBoardTensors = (trainBoardTensors - mean) / std

    testBoardTensors = boardTensors[N-10:N]
    testBoardTensors = (testBoardTensors - mean) / std
    ddpm = DDPM(UNet(), device=device)

    # optimizer
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-3)

    trainDS = torch.utils.data.TensorDataset(trainBoardTensors)
    trainDL = torch.utils.data.DataLoader(trainDS, batch_size=batch_size, shuffle=True)

    if loadModel:
        ddpm = torch.load(modelName)
    else:

        # training loop
        mse = torch.nn.MSELoss()
        for epoch in tqdm(range(epochs), desc=f"Training progress", colour="#00ff00"):
            epoch_loss = 0.0
            for step, batch in enumerate(tqdm(trainDL, leave=False, desc=f"Epoch {epoch + 1}/{epochs}", colour="#005500")):
                optimizer.zero_grad()
                # Loading data
                x0 = batch[0].to(device)

                eta = torch.randn_like(x0).to(device)
                t = torch.randint(0, ddpm.n_steps, (x0.size(0),)).to(device)

                noisy = ddpm(x0, t, eta)
                # predict the noise
                noisePred = ddpm.noisePred(noisy, t)
                # now calculate the loss
                loss = mse(noisePred, eta)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(x0) / len(trainDS)
            print(f"Epoch {epoch + 1}/{epochs} loss: {epoch_loss}")
        torch.save(ddpm, modelName)
    # lets test out the noising

    with torch.no_grad():
        print("Generating some boards")
        # start from pure noise for now
        ts = torch.zeros(10, dtype=torch.long).to(device) + ddpm.n_steps
        fake = ddpm.createFromNoise(10, ts, writeToGif=False)
        # reverse the z score on the fake
        fake = fake * std + mean
        for i in range(10):
            print("board %d" % i)
            board = fake[i, :]
            board = torch.round(board, decimals=0)
            # add an empty dimension in front
            board = board.unsqueeze(0)
            board = decodeBoard(board)

            hmm = chess.svg.board(board=board, size=600)
            # write this to a file
            with open("./GeneratedBoards/boardFromNoise%d.svg" % i, 'w') as f:
                f.write(hmm)

        # now lets create some new boards from an existing sample
        for i in range(10):
            board = testBoardTensors[i:i+1, :]
            # lets store the starting board as well
            boardStored = board * std + mean
            boardStored = torch.round(boardStored, decimals=0)
            boardStored = decodeBoard(boardStored)
            hmm = chess.svg.board(board=boardStored, size=600)
            # write this to a file
            with open("./GeneratedBoards/boardReal%d.svg" % i, 'w') as f:
                f.write(hmm)

            # make a long 1x1 tensor of 10
            t = torch.zeros(1, 1, dtype=torch.long).to(device) + 50
            board = ddpm.createFromReal(board, t)
            # reverse the z score on the fake
            board = board * std + mean
            board = torch.round(board, decimals=0)
            board = decodeBoard(board)
            hmm = chess.svg.board(board=board, size=600)
            # write this to a file
            with open("./GeneratedBoards/boardFromReal%d.svg" % i, 'w') as f:
                f.write(hmm)


        # ts = torch.zeros(10, dtype=torch.long).to(device)
        # noisy = ddpm(testBoardTensors, ts, None)
        # ddpm.noisePred(noisy, ts)
        # for i in range(10):
        #     noise = noisy[i, :]
        #     real = testBoardTensors[i, :]
        #     noise = torch.round(noise, decimals=0)
        #     real = torch.round(real, decimals=0)
        #     # add an empty dimension in front
        #     noise = noise.unsqueeze(0)
        #     real = real.unsqueeze(0)
        #     realBoard = decodeBoard(real)
        #     hmm = chess.svg.board(board=realBoard, size=400)
        #     # write this to a file
        #     with open("./GeneratedBoards/realboard%d.svg" % i, 'w') as f:
        #         f.write(hmm)
        #     noiseBoard = decodeBoard(noise)
        #     hmm = chess.svg.board(board=noiseBoard, size=400)
        #     # write this to a file
        #     with open("./GeneratedBoards/noiseboard%d.svg" % i, 'w') as f:
        #         f.write(hmm)


