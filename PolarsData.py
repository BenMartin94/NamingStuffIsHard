import polars as pl
import torch
import torch.utils.data
import numpy as np
import threading
import itertools

def parallel_range(id, nworkers, start, stop):
    
    local = (stop - start + 1) // nworkers
    remainder = (stop - start + 1) % nworkers
    
    istart = id*local + start + min(id, remainder)
    istop = istart + local
    if remainder > id:
        istop += 1
    return istart, istop

class PolarsDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, N, batch_size=1000) -> None:
        super().__init__()
        self.frame = pl.read_parquet(filepath)
        self.piecemap = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]
        # print(self.frame)
        self.N = N
        # mini batching in the dataset
        self.batchsize = batch_size
        self.offset = 0

    def __getitem__(self, index):
        # batchIndex = index % self.batchsize
        # create a new batch if needed
        # if batchIndex == 0:
        if self.offset + self.batchsize > self.N:
            self.offset = self.N - self.batchsize
        self.setupNextBatch()
        self.offset += self.batchsize

        if self.offset >= self.N:
            self.offset = 0
        
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


class PolarsDataStream(torch.utils.data.IterableDataset):
    def __init__(self, filename, N, batch_size=5*2048) -> None:
        super().__init__()
        self.lazyframe = pl.scan_parquet(filename, parallel='none')
        self.N = N
        self.batch_size=batch_size

        self.offset = 0
        self.index = 0
        
        self.piecemap = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]
        
        self.task = threading.Thread(target=self.setupNextBatch)
        self.task.start()
       
        
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info() 
        id = worker_info.id
        nworkers = worker_info.num_workers
        
        self.worker_batch_size = self.batch_size
        
        total = (self.N // self.worker_batch_size)
        start, stop = parallel_range(id, nworkers, 0, total)
        
        mapped_itr = map(self.getBatch, range(start, stop))
        
        return mapped_itr

    # def __next__(self):
    #     # out of data, get more
    #     data_index = self.index % self.batch_size
                
    #     if data_index == 0:
    #         self.task.join()
    #         self.boards = self.new_boards
    #         self.labels = self.new_labels
    #         self.lengths = self.new_lengths
            
    #         print(f"Dataset setting up new batch")
    #         self.offset += self.batch_size
    #         # wrap around
    #         if self.offset >= self.N:
    #             self.offset = 0
                
    #         self.task = threading.Thread(target=self.setupNextBatch)
    #         self.task.start()
           
            
    #     self.index += self.batch_size
        
    #     return self.boards, self.labels, self.lengths
	
    def getBatch(self, idx):
        offset = idx*self.worker_batch_size
        size = self.worker_batch_size
        
        frameSlice = self.lazyframe.slice(offset, size)#.collect()
        elos = frameSlice.select(["white_elo", "black_elo"]).collect()
        
        white_elo = elos["white_elo"].to_numpy(zero_copy_only=True)
        black_elo = elos["black_elo"].to_numpy(zero_copy_only=True)
        new_labels = torch.from_numpy(np.hstack([white_elo[:, None], black_elo[:, None]]))
        
        
		# postions evaluations
        win_chance = frameSlice.select(pl.col("win_chances")).collect()["win_chances"].to_list()
        list_of_evals = list(map(torch.tensor, win_chance))
        
		# create lengths this way
        lengths = list(map(len, list_of_evals))
        
        padded_evals = torch.nn.utils.rnn.pad_sequence(list_of_evals, batch_first=True)
        
        # print("worker getting games")
        boards = frameSlice.select(pl.col("games")).collect()["games"].to_list()
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
        print("worker finished getting batch")
        return conv_sequence.view(-1, maxLen, 18, 8, 8), new_labels, lengths
    
    def setupNextBatch(self):
        frameSlice = self.lazyframe.slice(self.offset, self.batch_size)#.collect()
        elos = frameSlice.select(["white_elo", "black_elo"]).collect()
        
        white_elo = elos["white_elo"].to_numpy(zero_copy_only=True)
        black_elo = elos["black_elo"].to_numpy(zero_copy_only=True)
        self.new_labels = torch.from_numpy(np.hstack([white_elo[:, None], black_elo[:, None]]))
        
        size = self.batch_size

        lengths = frameSlice.select(
            pl.col("win_chances").arr.lengths().alias("lengths")
        ).collect()["lengths"].to_list()
        
		# postions evaluations
        win_chance = frameSlice.select(pl.col("win_chances")).collect()["win_chances"].to_list()
        list_of_evals = list(map(torch.tensor, win_chance))
        padded_evals = torch.nn.utils.rnn.pad_sequence(list_of_evals, batch_first=True)
        
        boards = frameSlice.select(pl.col("games")).collect()["games"].to_list()
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
        
        self.new_boards = conv_sequence.view(-1, maxLen, 18, 8, 8)
        self.new_lengths = lengths   
	