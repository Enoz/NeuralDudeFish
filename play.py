import torch
import chess
import conversions
import copy

#Simple tree search with alpha beta pruning

def minimax(model, board, depth):
	return alphabeta(model, board, depth, -1000, 1000)
		
def alphabeta(model, board, depth, a, b):
	if board.is_game_over():
		if board.result() == "1-0":
			return None, 1
		elif board.result() == "0-1":
			return None, -1
		else:
			return None, 0.5
	if depth == 0:
		return None,model(conversions.board_to_onehot(board))[0].item()
	if board.turn == chess.WHITE:
		value = -1000
		best_move = None
		for move in board.legal_moves:
			child = copy.deepcopy(board)
			child.push(move)
			_, n_value = alphabeta(model, child, depth-1, a, b)
			if n_value > value:
				value = n_value
				best_move = move
			a = max(a, value)
			if a >= b:
				break
		return best_move, value
	else:
		value = 1000
		best_move = None
		for move in board.legal_moves:
			child = copy.deepcopy(board)
			child.push(move)
			_, n_value = alphabeta(model, child, depth-1, a, b)
			if n_value < value:
				value = n_value
				best_move = move
			b = min(b, value)
			if a >= b:
				break
		return best_move, value
			
			
		
