# NeuralDudeFish
Neural Network Chess Engine

#### Background
NeuralDudeFish is a chess engine using minimax tree search with a neural network evaluator. The  training is done by providing board positions, along with the end result of the game (win, lose, draw) to the neural network. 

#### Performance
Some sample games can be found [here](https://lichess.org/study/QFtj4Mf3 "here").

In these samples, the engine was trained on only 10,000 games (roughly 20 minutes of training) and performed poorly. It does not recognize threats, and seems to strictly stick to a small subset of opening moves with no regard to what the opponent plays.

I believe this is due to a lack of training samples, and perhaps can be revisited when the resources to train this engine for a longer period of time are available.

#### Setup
The script is ran using the following libraries:
- python-chess
- PyTorch

Sample games were human played, and downloaded from the [Lichess Game Database](https://database.lichess.org/)
The games should be placed in `trainpgn/`  and you should run `train.py` to produce a model.

You can then play agianst your model (as white) by running `play_as_white.py <model name>`
