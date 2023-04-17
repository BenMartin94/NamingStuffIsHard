import polars as pl
import torch
import torch.utils.data
from scipy.special import expit as sigmoid
import numpy as np
import pyarrow

torch.set_default_dtype(torch.float)

class PolarsDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, N, batch_size=1000) -> None:
        super().__init__()
        self.frame = pl.scan_parquet(filepath)
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

    def normalizationParams(self):
        elos = self.frame.select("white_elo", "black_elo")
        
        mean = elos.mean().collect()
        std = elos.std().collect()
        print(mean, std)
        
    def setupNextBatch(self):
        frameSlice = self.frame.slice(self.offset, self.batchsize).collect()
        print(frameSlice)
        white_elo = frameSlice["white_elo"].to_numpy(zero_copy_only=True)
        black_elo = frameSlice["black_elo"].to_numpy(zero_copy_only=True)
        self.labels = torch.from_numpy(np.hstack([white_elo[:, None], black_elo[:, None]]))
        
        size = frameSlice.height

        # postions evaluations
        evals = frameSlice["evals"]
        evals_idx = frameSlice["evals_idx"]
        mate_evals = frameSlice["mate_evals"]
        mate_evals_idx = frameSlice["mate_evals_idx"]
        
		# length of each sequence, needed to pack for pytorch
        frameSlice = frameSlice.with_columns(
            pl.struct(["evals", "mate_evals"]).apply(
            	lambda x: len(x["evals"]) + len("mate_evals")
            ).alias("eval_lengths")
        )
        print(frameSlice["eval_lengths"])
        
        winChance = frameSlice.select(
            pl.concat_list([
            	"evals_idx", 
            	"mate_evals_idx"
            ]).apply(lambda x: x-1).alias("all_idx"),
            pl.concat_list([
            	pl.col("evals").apply(sigmoid), # sigmoid to stockfish evaluation to get win probability
            	pl.col("mate_evals").apply(np.sign) # sign function to get win probability for mate in x: -100% or +100%
            ]).alias("win_chance"),
        )
        # sort combined winchances
        winChance = winChance.select(
            pl.col("win_chance").arr.take(pl.col("all_idx"))
		)
        print(winChance)
        exit()
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
            
        # the positions themselves
        pawns = frameSlice["pawns"]
        bishops = frameSlice["bishops"]
        knights = frameSlice["knights"]
        rooks = frameSlice["rooks"]
        queens = frameSlice["queens"]
        kings = frameSlice["kings"]
        white = frameSlice["white_mask"]
        black = frameSlice["black_mask"]

        
       

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

        