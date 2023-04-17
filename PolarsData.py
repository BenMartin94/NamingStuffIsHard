import polars
import torch
import torch.utils.data
from scipy.special import expit as sigmoid
import numpy as np

torch.set_default_dtype(torch.float)

class PolarsDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, N, batch_size=1000) -> None:
        super().__init__()
        self.frame = polars.scan_parquet(filepath)
        self.N = N // batch_size
        # mini batching in the dataset
        self.batchsize = batch_size
        self.offset = 0

    def __getitem__(self, index):
        # batchIndex = index % self.batchsize
        # create a new batch if needed
        # if batchIndex == 0:
        self.setupNextBatch()
        self.offset += self.batchsize

        return (
            self.boards,
            self.labels,
            self.lengths,
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
            rawBoard = torch.zeros(seqlen, 13, 8, 8)

            board = rawBoard.view(seqlen, 13, 64)

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

			# win chance
            board[:, 12, :] = self.winChances[i][:,None]
            self.boards.append(rawBoard)

        # labels
        white_elo = torch.from_numpy(white_elo.to_numpy())[:, None]
        black_elo = torch.from_numpy(black_elo.to_numpy())[:, None]

        self.labels = torch.cat([white_elo, black_elo], dim=1).float()
        