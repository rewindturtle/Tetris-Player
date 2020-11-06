import os
from Tetris_Player import Tetris_Player


MAIN_DIRECTORY = os.path.dirname(__file__)
GAME_PATH = os.path.join(MAIN_DIRECTORY, 'Game')
STATE_PATH = os.path.join(GAME_PATH, 'Level9A.state')


if __name__ == '__main__':
    player = Tetris_Player(game_path=GAME_PATH,
                           state_path=STATE_PATH,
                           game_speed=0,
                           render=True)
    player.play()
