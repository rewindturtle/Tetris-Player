# Tetris-Player
An algorithm that is capable of acheiving the max highscore in Tetris for the Nintendo Entertainment System. Based on the paper "Why Most Decisions Are Easy in Tetris—And Perhaps in Other Sequential Decision Problems, As Well" by Ozgur Simsek, Simon Algorta, and Amit Kothiyal.

A frame from the game is used to determine the current board configuration, current tetrimino, and next tetrimino. A score is then assigned to each possible block placement using the current and next tetrimino. The placement with the highest score is chosen as the next move. The score is determined by calculating eight parameters of the board and multiplying them by a weight. The parameters are taken from the paper "Why Most Decisions Are Easy in Tetris—And Perhaps in Other Sequential Decision Problems, As Well" by Ozgur Simsek, Simon Algorta, and Amit Kothiyal. 

Scoring Parameters:
- Number of rows with holes
- Number of total holes
- How much the piece increases the height of the board when placed
- How many blocks are above each hole
- The number of times the board transistions from a filled to empty block as one moves from left to right through each row
- Number of "wells" (areas of the board that have a block on each side but none above
- Number lines cleared by placing a block

The weighting for each parameter is determined through a cross-entropy algorithm. THe game is played through gym-retro, a python library by Open-Ai that emulates videogames in python. The state and data files are included in the repo. However, the ROM of the game is not.


Link to paper: http://proceedings.mlr.press/v48/simsek16.html

Link to Gym Retro: https://openai.com/blog/gym-retro/
