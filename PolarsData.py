import polars as pl
import torch
import torch.utils.data
from scipy.special import expit as sigmoid
import numpy as np

# torch.set_default_dtype(torch.float)


class PolarsDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, N, batch_size=1000) -> None:
        super().__init__()
        self.frame = pl.read_parquet(filepath)
        self.piecemap = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]
        # print(self.frame)
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
        frameSlice = self.frame.slice(self.offset, self.batchsize)# .collect()
        white_elo = frameSlice["white_elo"].to_numpy(zero_copy_only=True)
        black_elo = frameSlice["black_elo"].to_numpy(zero_copy_only=True)
        self.labels = torch.from_numpy(np.hstack([white_elo[:, None], black_elo[:, None]]))
        
        size = frameSlice.height

        lengths = frameSlice.select(
            pl.col("win_chances").arr.lengths().alias("lengths")
        )["lengths"].to_list()
        
		# postions evaluations
        win_chance = frameSlice["win_chances"].to_list()
        list_of_evals = list(map(torch.tensor, win_chance))
        padded_evals = torch.nn.utils.rnn.pad_sequence(list_of_evals, batch_first=True)
        
        boards = frameSlice["games"].to_list()
        list_of_tensors = list(map(torch.tensor, boards))
        padded_boards = torch.nn.utils.rnn.pad_sequence(list_of_tensors, batch_first=True)
        
        maxLen = padded_boards.shape[1]
        # convert to convolutional format
        conv_sequence = torch.zeros((size, maxLen, 18, 64))
        
        for i in range(len(self.piecemap)):
            conv_sequence[...,i,:] = padded_boards[...,0:64] == self.piecemap[i]

		# 12->encoding of white move or not
		# 13->white kingside castle
		# 14->white queenside castle
		# 15->black kindside castle
		# 16->black queenside castle
        conv_sequence[..., 12:17, :] = padded_boards[..., 64:69, None]
        
		#17->movescore
        conv_sequence[..., 17, :] = padded_evals[..., :, None]
        
        self.boards = conv_sequence.view(-1, maxLen, 18, 8, 8)
        self.labels
        self.lengths = lengths

		