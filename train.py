import chess.pgn
import os
import conversions
import chess_model

import torch

#folder containing pgn list
SAMPLE_FOLDER = "trainpgn/"

#where to save model
SAVE_LOCATION = "model10000.torch"

#number of games to train on
MAXGAMES = 10000

#max epochs if we run out of games
EPOCHS = 1

model = chess_model.make_model()

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4

#games trained on
num = 1
for epoch in range(EPOCHS):
	for filename in os.listdir(SAMPLE_FOLDER):
		if not filename.endswith(".pgn"):
			continue
		pgn = open(SAMPLE_FOLDER + filename)
		game = chess.pgn.read_game(pgn)
		while game != None:
			num = num + 1
			if num%100 == 0:
				print("game #%i" % num)
			if num>=MAXGAMES:
				break;
			
			y = 0.5
			if game.headers['Result']=='1-0':
				y = 1
			elif game.headers['Result'] == '0-1':
				y = 0
			y = torch.FloatTensor([y])
			board = game.board()
			for move in game.mainline_moves():
				board.push(move)

				y_pred = model(conversions.board_to_onehot(board))
				loss = loss_fn(y_pred, y)
				model.zero_grad()
				loss.backward()
				
				with torch.no_grad():
					for param in model.parameters():
						param -= learning_rate * param.grad
			game = chess.pgn.read_game(pgn)
		if num>=MAXGAMES:
			break;
			
print("Saving model...")
torch.save(model.state_dict(), SAVE_LOCATION)
print("Model saved at " + SAVE_LOCATION)
