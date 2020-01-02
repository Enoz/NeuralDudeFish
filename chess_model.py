import torch
import conversions
import chess

# 3 Hidden layers with ReLU Activation
# Output layer sent through sigmoid

def make_model():
	input_size = len(conversions.board_to_onehot(chess.Board()))
	model = torch.nn.Sequential(
		torch.nn.Linear(input_size, 1048),
		torch.nn.ReLU(),
		torch.nn.Linear(1048,500),
		torch.nn.ReLU(),
		torch.nn.Linear(500,50),
		torch.nn.ReLU(),
		torch.nn.Linear(50,1),
		torch.nn.Sigmoid(),
	)
	return model
