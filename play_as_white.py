#Simple driver to player as white against the engine

import play
import torch
import chess_model
import conversions
import chess
import sys

if (len(sys.argv) != 2):
	print("Usage: %s <model path>" % sys.argv[0])
	quit()
	
model_path = sys.argv[1]
model = chess_model.make_model()
model.load_state_dict(torch.load(model_path))

board = chess.Board()

while not board.is_game_over():
	if(board.turn == chess.WHITE):
		print(board)
		mv = chess.Move.from_uci(input("Enter a move: "))
		if mv in board.legal_moves:
			board.push(mv)
	if(board.turn == chess.BLACK):
		mv,val = play.alphabeta(model, board, 5)
		print(str(mv) + " With " + str(val) + " eval")
		board.push(mv)
		
