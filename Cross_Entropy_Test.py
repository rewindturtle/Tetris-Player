import numpy as np
import os
from CEM_Player import Tetris_Player


MAIN_DIRECTORY = os.path.dirname(__file__)
GAME_PATH = os.path.join(MAIN_DIRECTORY, 'Game')
STATE_PATH = os.path.join(GAME_PATH, 'Level9A.state')


NUM_PLAYERS = 25
NUM_ELITE_PLAYERS = 5
NUM_GENERATIONS = 100
NUM_GAMES = 4
MAX_LINES = 100
MIN_STD = 0.01


if __name__ == '__main__':
    means = [-2.404, -1.977, -1.308, -1.263, -1.049, -0.922, 0.66, -0.161]
    stds = [1, 1, 1, 1, 1, 1, 1, 1]
    best_weights = [-2.404, -1.977, -1.308, -1.263, -1.049, -0.922, 0.66, -0.161]
    high_score = 0
    player = Tetris_Player(game_path=GAME_PATH,
                           state_path=STATE_PATH,
                           max_lines=MAX_LINES,
                           render=False)
    for i in range(NUM_GENERATIONS):
        print("Best Weights: {}".format(best_weights))
        print("New Means: {}".format(means))
        print("New Stds: {}".format(stds))
        print("Generation {}".format(i))
        weights = []
        scores = []
        for j in range(NUM_PLAYERS):
            print("_________________________________________________________")
            print("Player {}".format(j))
            new_weights = np.random.normal(means, stds)
            new_weights = np.clip(new_weights,
                                  [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, -np.inf],
                                  [0, 0, 0, np.inf, np.inf, 0, np.inf, np.inf])
            weights.append(new_weights)
            player.change_weights(new_weights)
            print("Weights: {}".format(new_weights))
            score = player.play_multiple_games(num_games=NUM_GAMES)
            if score > high_score:
                high_score = score
                best_weights = new_weights
            print("Average Score: {}".format(score))
            scores.append(score)

        elite_idx = np.asarray(scores).argsort()[-NUM_ELITE_PLAYERS:][::-1]
        print("***************************************************************")
        elite_weights = []
        for idx in elite_idx:
            elite_weights.append(weights[idx])
        means = np.asarray(elite_weights).mean(axis=0)
        stds = np.asarray(elite_weights).std(axis=0)
        stds = np.clip(stds, a_min=MIN_STD, a_max=None)
        print("/////////////////////////////////////////////////////////////////////")
