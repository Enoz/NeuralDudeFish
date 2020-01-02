import chess
import torch

# Onehot encoding of the chess board for Neural Network input
# Each square has 12 bits, 2 for each piece to account for color

def board_to_onehot(board):
	il = []
	
	il.append(board.turn == chess.WHITE and 1 or 0)
	for square in chess.SQUARES:
		rep = [0]*12
		piece = board.piece_at(square)
		if piece != None:
			color = piece.color
			idx = piece.piece_type-1
			if color == chess.BLACK:
				idx *= 2
			rep[idx] = 1
		for bit in rep:
			il.append(bit)
	return torch.FloatTensor(il)
	
