def loadKaggleData(path):
    import pandas as pd
    import torch
    import sys
    
    neginf = -sys.maxsize - 1
    
    N = 25000
    
    df = pd.read_pickle(path)

    boardTensors = list(map(torch.Tensor, df["boards"][0:N]))
    moveTensors = list(map(torch.Tensor, df["MoveScores"][0:N]))
    
    # exclude games that are missing evals
    keptGames = [~torch.any(t == neginf) for t in moveTensors]
    boardTensors = [t for (t, mask) in zip(boardTensors, keptGames) if mask]
    moveTensors = [t for (t, mask) in zip(moveTensors, keptGames) if mask]
    
    lengths = torch.tensor(list(map(len, boardTensors)))

    moves = torch.nn.utils.rnn.pad_sequence(moveTensors, batch_first=True)
    boards = torch.nn.utils.rnn.pad_sequence(boardTensors, batch_first=True)
    
    # hack to correct move eval scores to new version of stockfish
    moves = torch.nn.functional.sigmoid(moves / 100)
    
    X = torch.cat([boards, moves.unsqueeze(2)],dim=2).float()
    
	# elos
    white_elo = torch.tensor(df["white_elo"][0:N])[keptGames]
    black_elo = torch.tensor(df["black_elo"][0:N])[keptGames]
    y = torch.cat([white_elo.unsqueeze(1), black_elo.unsqueeze(1)], dim=1).float()
    
    return X, y, lengths
    
