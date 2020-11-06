import numpy as np
import retro
import cv2


J_BLOCK = 1
L_BLOCK = 2
T_BLOCK = 3
Z_BLOCK = 4
S_BLOCK = 5
O_BLOCK = 6
I_BLOCK = 7

ROT_0 = 0
ROT_90 = 1
ROT_180 = 2
ROT_270 = 3

NO_ACTION = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0])
START = np.asarray([0, 0, 0, 1, 0, 0, 0, 0, 0])
RIGHT = np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 0])
LEFT = np.asarray([0, 0, 0, 0, 0, 0, 1, 0, 0])
A = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1])
B = np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0])


def _format_frame(raw_frame):
    current_block = _find_current_block(raw_frame)
    next_block = _find_next_block(raw_frame)
    board = raw_frame[39:201, 87:168, :]
    board = cv2.resize(board, (10, 20), interpolation=cv2.INTER_CUBIC)
    board = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY)
    board = board[4:20, :]
    board = np.where(board > 0, 1, 0)
    return np.flipud(board), current_block, next_block


def _find_current_block(raw_frame):
    frame = raw_frame[40:55, 112:143]
    frame = cv2.resize(frame, (4, 2), interpolation=cv2.INTER_CUBIC)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = np.where(frame > 0, 1, 0)
    frame = frame.flatten()
    if np.array_equal(frame, [0, 1, 1, 1, 0, 0, 0, 1]):
        return J_BLOCK
    elif np.array_equal(frame, [0, 1, 1, 1, 0, 1, 0, 0]):
        return L_BLOCK
    elif np.array_equal(frame, [0, 1, 1, 1, 0, 0, 1, 0]):
        return T_BLOCK
    elif np.array_equal(frame, [0, 1, 1, 0, 0, 0, 1, 1]):
        return Z_BLOCK
    elif np.array_equal(frame, [0, 0, 1, 1, 0, 1, 1, 0]):
        return S_BLOCK
    elif np.array_equal(frame, [0, 1, 1, 0, 0, 1, 1, 0]):
        return O_BLOCK
    elif np.array_equal(frame, [1, 1, 1, 1, 0, 0, 0, 0]):
        return I_BLOCK
    else:
        return 0


def _find_next_block(raw_frame):
    frame = raw_frame[112:127, 184:215]
    frame = cv2.resize(frame, (4, 2), interpolation=cv2.INTER_CUBIC)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = np.where(frame > 0, 1, 0)
    frame = frame.flatten()
    if np.array_equal(frame, [1, 1, 1, 1, 0, 0, 0, 1]):
        return J_BLOCK
    elif np.array_equal(frame, [1, 1, 1, 1, 1, 0, 0, 0]):
        return L_BLOCK
    elif np.array_equal(frame, [1, 1, 1, 1, 0, 1, 1, 0]):
        return T_BLOCK
    elif np.array_equal(frame, [1, 1, 1, 0, 0, 1, 1, 1]):
        return Z_BLOCK
    elif np.array_equal(frame, [0, 1, 1, 1, 1, 1, 1, 0]):
        return S_BLOCK
    elif np.array_equal(frame, [0, 1, 1, 0, 0, 1, 1, 0]):
        return O_BLOCK
    elif np.array_equal(frame, [1, 1, 1, 1, 1, 1, 1, 1]):
        return I_BLOCK
    else:
        return 0


def _add_block(board, col_height, block, col, rot):
    done = False
    next_board = board.copy()
    # PLACING J BLOCK
    if block == J_BLOCK:
        if rot == ROT_0:
            if col == 0:
                heights = [col_height[0], col_height[1], col_height[2]]
                height = np.max(heights)
                pos = int(np.argmax(heights))
                if heights[pos] > heights[2]:
                    if height > 15:
                        done = True
                    else:
                        next_board[height][0] = 1
                        next_board[height][1] = 1
                        next_board[height - 1][2] = 1
                        next_board[height][2] = 1
                else:
                    if height + 2 > 15:
                        done = True
                    else:
                        next_board[height + 1][0] = 1
                        next_board[height + 1][1] = 1
                        next_board[height][2] = 1
                        next_board[height + 1][2] = 1
            elif col == 9:
                heights = [col_height[7], col_height[8], col_height[9]]
                height = np.max(heights)
                pos = int(np.argmax(heights))
                if heights[pos] > heights[2]:
                    if height > 15:
                        done = True
                    else:
                        next_board[height][7] = 1
                        next_board[height][8] = 1
                        next_board[height - 1][9] = 1
                        next_board[height][9] = 1
                else:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height + 1][7] = 1
                        next_board[height + 1][8] = 1
                        next_board[height][9] = 1
                        next_board[height + 1][9] = 1
            else:
                heights = [col_height[col - 1], col_height[col], col_height[col + 1]]
                height = np.max(heights)
                pos = int(np.argmax(heights))
                if heights[pos] > heights[2]:
                    if height > 15:
                        done = True
                    else:
                        next_board[height][col - 1] = 1
                        next_board[height][col] = 1
                        next_board[height - 1][col + 1] = 1
                        next_board[height][col + 1] = 1
                else:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height + 1][col - 1] = 1
                        next_board[height + 1][col] = 1
                        next_board[height][col + 1] = 1
                        next_board[height + 1][col + 1] = 1
        elif rot == ROT_90:
            if col == 0:
                height = max(col_height[0], col_height[1])
                if height + 2 > 15:
                    done = True
                else:
                    next_board[height][0] = 1
                    next_board[height][1] = 1
                    next_board[height + 1][1] = 1
                    next_board[height + 2][1] = 1
            else:
                height = max(col_height[col - 1], col_height[col])
                if height + 2 > 15:
                    done = True
                else:
                    next_board[height][col - 1] = 1
                    next_board[height][col] = 1
                    next_board[height + 1][col] = 1
                    next_board[height + 2][col] = 1
        elif rot == ROT_180:
            if col == 0:
                height = max(col_height[0], col_height[1], col_height[2])
                if height + 1 > 15:
                    done = True
                else:
                    next_board[height][0] = 1
                    next_board[height + 1][0] = 1
                    next_board[height][1] = 1
                    next_board[height][2] = 1
            elif col == 9:
                height = max(col_height[7], col_height[8], col_height[9])
                if height + 1 > 15:
                    done = True
                else:
                    next_board[height][7] = 1
                    next_board[height + 1][7] = 1
                    next_board[height][8] = 1
                    next_board[height][9] = 1
            else:
                height = max(col_height[col - 1], col_height[col], col_height[col + 1])
                if height + 1 > 15:
                    done = True
                else:
                    next_board[height][col - 1] = 1
                    next_board[height + 1][col - 1] = 1
                    next_board[height][col] = 1
                    next_board[height][col + 1] = 1
        elif rot == ROT_270:
            if col == 9:
                height = max(col_height[8], col_height[9])
                if col_height[9] > col_height[8] + 1:
                    if height > 15:
                        done = True
                    else:
                        next_board[height - 2][8] = 1
                        next_board[height - 1][8] = 1
                        next_board[height][8] = 1
                        next_board[height][9] = 1
                elif col_height[9] > col_height[8]:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height - 1][8] = 1
                        next_board[height][8] = 1
                        next_board[height + 1][8] = 1
                        next_board[height + 1][9] = 1
                else:
                    if height + 2 > 15:
                        done = True
                    else:
                        next_board[height][8] = 1
                        next_board[height + 1][8] = 1
                        next_board[height + 2][8] = 1
                        next_board[height + 2][9] = 1
            else:
                height = max(col_height[col], col_height[col + 1])
                if col_height[col + 1] > col_height[col] + 1:
                    if height > 15:
                        done = True
                    else:
                        next_board[height - 2][col] = 1
                        next_board[height - 1][col] = 1
                        next_board[height][col] = 1
                        next_board[height][col + 1] = 1
                elif col_height[col + 1] > col_height[col]:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height - 1][col] = 1
                        next_board[height][col] = 1
                        next_board[height + 1][col] = 1
                        next_board[height + 1][col + 1] = 1
                else:
                    if height + 2 > 15:
                        done = True
                    else:
                        next_board[height][col] = 1
                        next_board[height + 1][col] = 1
                        next_board[height + 2][col] = 1
                        next_board[height + 2][col + 1] = 1
    # PLACING L BLOCK
    elif block == L_BLOCK:
        if rot == ROT_0:
            if col == 0:
                heights = [col_height[0], col_height[1], col_height[2]]
                height = np.max(heights)
                pos = int(np.argmax(heights))
                if heights[pos] > heights[0]:
                    if height > 15:
                        done = True
                    else:
                        next_board[height - 1][0] = 1
                        next_board[height][0] = 1
                        next_board[height][1] = 1
                        next_board[height][2] = 1
                else:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height][0] = 1
                        next_board[height + 1][0] = 1
                        next_board[height + 1][1] = 1
                        next_board[height + 1][2] = 1
            elif col == 9:
                heights = [col_height[7], col_height[8], col_height[9]]
                height = np.max(heights)
                pos = int(np.argmax(heights))
                if heights[pos] > heights[0]:
                    if height > 15:
                        done = True
                    else:
                        next_board[height - 1][7] = 1
                        next_board[height][7] = 1
                        next_board[height][8] = 1
                        next_board[height][9] = 1
                else:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height][7] = 1
                        next_board[height + 1][7] = 1
                        next_board[height + 1][8] = 1
                        next_board[height + 1][9] = 1
            else:
                heights = [col_height[col - 1], col_height[col], col_height[col + 1]]
                height = np.max(heights)
                pos = int(np.argmax(heights))
                if heights[pos] > heights[0]:
                    if height > 15:
                        done = True
                    else:
                        next_board[height - 1][col - 1] = 1
                        next_board[height][col - 1] = 1
                        next_board[height][col] = 1
                        next_board[height][col + 1] = 1
                else:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height][col - 1] = 1
                        next_board[height + 1][col - 1] = 1
                        next_board[height + 1][col] = 1
                        next_board[height + 1][col + 1] = 1
        elif rot == ROT_90:
            if col == 0:
                height = max(col_height[0], col_height[1])
                if col_height[0] > col_height[1] + 1:
                    if height > 15:
                        done = True
                    else:
                        next_board[height][0] = 1
                        next_board[height - 2][1] = 1
                        next_board[height - 1][1] = 1
                        next_board[height][1] = 1
                elif col_height[0] > col_height[1]:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height + 1][0] = 1
                        next_board[height - 1][1] = 1
                        next_board[height][1] = 1
                        next_board[height + 1][1] = 1
                else:
                    if height + 2 > 15:
                        done = True
                    else:
                        next_board[height + 2][0] = 1
                        next_board[height][1] = 1
                        next_board[height + 1][1] = 1
                        next_board[height + 2][1] = 1
            else:
                height = max(col_height[col - 1], col_height[col])
                if col_height[col - 1] > col_height[col] + 1:
                    if height > 15:
                        done = True
                    else:
                        next_board[height][col - 1] = 1
                        next_board[height - 2][col] = 1
                        next_board[height - 1][col] = 1
                        next_board[height][col] = 1
                elif col_height[col - 1] > col_height[col]:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height + 1][col - 1] = 1
                        next_board[height - 1][col] = 1
                        next_board[height][col] = 1
                        next_board[height + 1][col] = 1
                else:
                    if height + 2 > 15:
                        done = True
                    else:
                        next_board[height + 2][col - 1] = 1
                        next_board[height][col] = 1
                        next_board[height + 1][col] = 1
                        next_board[height + 2][col] = 1
        elif rot == ROT_180:
            if col == 0:
                height = max(col_height[0], col_height[1], col_height[2])
                if height + 1 > 15:
                    done = True
                else:
                    next_board[height][0] = 1
                    next_board[height][1] = 1
                    next_board[height][2] = 1
                    next_board[height + 1][2] = 1
            elif col == 9:
                height = max(col_height[7], col_height[8], col_height[9])
                if height + 1 > 15:
                    done = True
                else:
                    next_board[height][7] = 1
                    next_board[height][8] = 1
                    next_board[height][9] = 1
                    next_board[height + 1][9] = 1
            else:
                height = max(col_height[col - 1], col_height[col], col_height[col + 1])
                if height + 1 > 15:
                    done = True
                else:
                    next_board[height][col - 1] = 1
                    next_board[height][col] = 1
                    next_board[height][col + 1] = 1
                    next_board[height + 1][col + 1] = 1
        elif rot == ROT_270:
            if col == 9:
                height = max(col_height[8], col_height[9])
                if height + 2 > 15:
                    done = True
                else:
                    next_board[height][8] = 1
                    next_board[height + 1][8] = 1
                    next_board[height + 2][8] = 1
                    next_board[height][9] = 1
            else:
                height = max(col_height[col], col_height[col + 1])
                if height + 2 > 15:
                    done = True
                else:
                    next_board[height][col] = 1
                    next_board[height + 1][col] = 1
                    next_board[height + 2][col] = 1
                    next_board[height][col + 1] = 1
    # PLACING T BLOCK
    elif block == T_BLOCK:
        if rot == ROT_180:
            if col == 0:
                height = max(col_height[0], col_height[1], col_height[2])
                if height + 1 > 15:
                    done = True
                else:
                    next_board[height][0] = 1
                    next_board[height][1] = 1
                    next_board[height + 1][1] = 1
                    next_board[height][2] = 1
            elif col == 9:
                height = max(col_height[7], col_height[8], col_height[9])
                if height + 1 > 15:
                    done = True
                else:
                    next_board[height][7] = 1
                    next_board[height][8] = 1
                    next_board[height + 1][8] = 1
                    next_board[height][9] = 1
            else:
                height = max(col_height[col - 1], col_height[col], col_height[col + 1])
                if height + 1 > 15:
                    done = True
                else:
                    next_board[height][col - 1] = 1
                    next_board[height][col] = 1
                    next_board[height + 1][col] = 1
                    next_board[height][col + 1] = 1
        elif rot == ROT_90:
            if col == 0:
                height = max(col_height[0], col_height[1])
                if col_height[0] > col_height[1]:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height][0] = 1
                        next_board[height - 1][1] = 1
                        next_board[height][1] = 1
                        next_board[height + 1][1] = 1
                else:
                    if height + 2 > 15:
                        done = True
                    else:
                        next_board[height + 1][0] = 1
                        next_board[height][1] = 1
                        next_board[height + 1][1] = 1
                        next_board[height + 2][1] = 1
            else:
                height = max(col_height[col - 1], col_height[col])
                if col_height[col - 1] > col_height[col]:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height][col - 1] = 1
                        next_board[height - 1][col] = 1
                        next_board[height][col] = 1
                        next_board[height + 1][col] = 1
                else:
                    if height + 2 > 15:
                        done = True
                    else:
                        next_board[height + 1][col - 1] = 1
                        next_board[height][col] = 1
                        next_board[height + 1][col] = 1
                        next_board[height + 2][col] = 1
        elif rot == ROT_0:
            if col == 0:
                heights = [col_height[0], col_height[1], col_height[2]]
                height = np.max(heights)
                pos = int(np.argmax(heights))
                if heights[pos] > heights[1]:
                    if height > 15:
                        done = True
                    else:
                        next_board[height][0] = 1
                        next_board[height - 1][1] = 1
                        next_board[height][1] = 1
                        next_board[height][2] = 1
                else:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height + 1][0] = 1
                        next_board[height][1] = 1
                        next_board[height + 1][1] = 1
                        next_board[height + 1][2] = 1
            elif col == 9:
                heights = [col_height[7], col_height[8], col_height[9]]
                height = np.max(heights)
                pos = int(np.argmax(heights))
                if heights[pos] > heights[1]:
                    if height > 15:
                        done = True
                    else:
                        next_board[height][7] = 1
                        next_board[height - 1][8] = 1
                        next_board[height][8] = 1
                        next_board[height][9] = 1
                else:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height + 1][7] = 1
                        next_board[height][8] = 1
                        next_board[height + 1][8] = 1
                        next_board[height + 1][9] = 1
            else:
                heights = [col_height[col - 1], col_height[col], col_height[col + 1]]
                height = np.max(heights)
                pos = int(np.argmax(heights))
                if heights[pos] > heights[1]:
                    if height > 15:
                        done = True
                    else:
                        next_board[height][col - 1] = 1
                        next_board[height - 1][col] = 1
                        next_board[height][col] = 1
                        next_board[height][col + 1] = 1
                else:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height + 1][col - 1] = 1
                        next_board[height][col] = 1
                        next_board[height + 1][col] = 1
                        next_board[height + 1][col + 1] = 1
        elif rot == ROT_270:
            if col == 9:
                height = max(col_height[8], col_height[9])
                if col_height[9] > col_height[8]:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height][9] = 1
                        next_board[height - 1][8] = 1
                        next_board[height][8] = 1
                        next_board[height + 1][8] = 1
                else:
                    if height + 2 > 15:
                        done = True
                    else:
                        next_board[height + 1][9] = 1
                        next_board[height][8] = 1
                        next_board[height + 1][8] = 1
                        next_board[height + 2][8] = 1
            else:
                height = max(col_height[col], col_height[col + 1])
                if col_height[col + 1] > col_height[col]:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height][col + 1] = 1
                        next_board[height - 1][col] = 1
                        next_board[height][col] = 1
                        next_board[height + 1][col] = 1
                else:
                    if height + 2 > 15:
                        done = True
                    else:
                        next_board[height + 1][col + 1] = 1
                        next_board[height][col] = 1
                        next_board[height + 1][col] = 1
                        next_board[height + 2][col] = 1
    # PLACING Z BLOCK
    elif block == Z_BLOCK:
        if rot == ROT_0 or rot == ROT_180:
            if col == 0:
                height = max(col_height[0], col_height[1], col_height[2])
                if (col_height[0] > col_height[1]) and (col_height[0] > col_height[2]):
                    if height > 15:
                        done = True
                    else:
                        next_board[height][0] = 1
                        next_board[height - 1][1] = 1
                        next_board[height][1] = 1
                        next_board[height - 1][2] = 1
                else:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height + 1][0] = 1
                        next_board[height][1] = 1
                        next_board[height + 1][1] = 1
                        next_board[height][2] = 1
            elif col == 9:
                height = max(col_height[7], col_height[8], col_height[9])
                if (col_height[7] > col_height[8]) and (col_height[7] > col_height[9]):
                    if height > 15:
                        done = True
                    else:
                        next_board[height][7] = 1
                        next_board[height - 1][8] = 1
                        next_board[height][8] = 1
                        next_board[height - 1][9] = 1
                else:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height + 1][7] = 1
                        next_board[height][8] = 1
                        next_board[height + 1][8] = 1
                        next_board[height][9] = 1
            else:
                height = max(col_height[col - 1], col_height[col], col_height[col + 1])
                if (col_height[col - 1] > col_height[col]) and (col_height[col - 1] > col_height[col + 1]):
                    if height > 15:
                        done = True
                    else:
                        next_board[height][col - 1] = 1
                        next_board[height - 1][col] = 1
                        next_board[height][col] = 1
                        next_board[height - 1][col + 1] = 1
                else:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height + 1][col - 1] = 1
                        next_board[height][col] = 1
                        next_board[height + 1][col] = 1
                        next_board[height][col + 1] = 1
        elif rot == ROT_90 or rot == ROT_270:
            if col == 9:
                height = max(col_height[8], col_height[9])
                if col_height[9] > col_height[8]:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height - 1][8] = 1
                        next_board[height][8] = 1
                        next_board[height][9] = 1
                        next_board[height + 1][9] = 1
                else:
                    if height + 2 > 15:
                        done = True
                    else:
                        next_board[height][8] = 1
                        next_board[height + 1][8] = 1
                        next_board[height + 1][9] = 1
                        next_board[height + 2][9] = 1
            else:
                height = max(col_height[col], col_height[col + 1])
                if col_height[col + 1] > col_height[col]:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height - 1][col] = 1
                        next_board[height][col] = 1
                        next_board[height][col + 1] = 1
                        next_board[height + 1][col + 1] = 1
                else:
                    if height + 2 > 15:
                        done = True
                    else:
                        next_board[height][col] = 1
                        next_board[height + 1][col] = 1
                        next_board[height + 1][col + 1] = 1
                        next_board[height + 2][col + 1] = 1
    # PLACING S BLOCK
    elif block == S_BLOCK:
        if rot == ROT_0 or rot == ROT_180:
            if col == 0:
                height = max(col_height[0], col_height[1], col_height[2])
                if (col_height[2] > col_height[0]) and (col_height[2] > col_height[1]):
                    if height > 15:
                        done = True
                    else:
                        next_board[height - 1][0] = 1
                        next_board[height - 1][1] = 1
                        next_board[height][1] = 1
                        next_board[height][2] = 1
                else:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height][0] = 1
                        next_board[height][1] = 1
                        next_board[height + 1][1] = 1
                        next_board[height + 1][2] = 1
            elif col == 9:
                height = max(col_height[7], col_height[8], col_height[9])
                if (col_height[9] > col_height[7]) and (col_height[9] > col_height[8]):
                    if height > 15:
                        done = True
                    else:
                        next_board[height - 1][7] = 1
                        next_board[height - 1][8] = 1
                        next_board[height][8] = 1
                        next_board[height][9] = 1
                else:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height][7] = 1
                        next_board[height][8] = 1
                        next_board[height + 1][8] = 1
                        next_board[height + 1][9] = 1
            else:
                height = max(col_height[col - 1], col_height[col], col_height[col + 1])
                if (col_height[col + 1] > col_height[col - 1]) and (col_height[col + 1] > col_height[col]):
                    if height > 15:
                        done = True
                    else:
                        next_board[height - 1][col - 1] = 1
                        next_board[height - 1][col] = 1
                        next_board[height][col] = 1
                        next_board[height][col + 1] = 1
                else:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height][col - 1] = 1
                        next_board[height][col] = 1
                        next_board[height + 1][col] = 1
                        next_board[height + 1][col + 1] = 1
        elif rot == ROT_90 or rot == ROT_270:
            if col == 9:
                height = max(col_height[8], col_height[9])
                if col_height[8] > col_height[9]:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height][8] = 1
                        next_board[height + 1][8] = 1
                        next_board[height][9] = 1
                        next_board[height - 1][9] = 1
                else:
                    if height + 2 > 15:
                        done = True
                    else:
                        next_board[height + 1][8] = 1
                        next_board[height + 2][8] = 1
                        next_board[height][9] = 1
                        next_board[height + 1][9] = 1
            else:
                height = max(col_height[col], col_height[col + 1])
                if col_height[col] > col_height[col + 1]:
                    if height + 1 > 15:
                        done = True
                    else:
                        next_board[height][col] = 1
                        next_board[height + 1][col] = 1
                        next_board[height][col + 1] = 1
                        next_board[height - 1][col + 1] = 1
                else:
                    if height + 2 > 15:
                        done = True
                    else:
                        next_board[height + 1][col] = 1
                        next_board[height + 2][col] = 1
                        next_board[height][col + 1] = 1
                        next_board[height + 1][col + 1] = 1
    # PLACING O BLOCK
    elif block == O_BLOCK:
        if col == 0:
            height = max(col_height[0], col_height[1])
            if height + 1 > 15:
                done = True
            else:
                next_board[height][0] = 1
                next_board[height + 1][0] = 1
                next_board[height][1] = 1
                next_board[height + 1][1] = 1
        else:
            height = max(col_height[col - 1], col_height[col])
            if height + 1 > 15:
                done = True
            else:
                next_board[height][col - 1] = 1
                next_board[height + 1][col - 1] = 1
                next_board[height][col] = 1
                next_board[height + 1][col] = 1
    # PLACING I BLOCK
    elif block == I_BLOCK:
        if rot == ROT_0 or rot == ROT_180:
            if col == 0 or col == 1:
                height = max(col_height[0], col_height[1], col_height[2], col_height[3])
                if height > 15:
                    done = True
                else:
                    next_board[height][0] = 1
                    next_board[height][1] = 1
                    next_board[height][2] = 1
                    next_board[height][3] = 1
            elif col == 9:
                height = max(col_height[6], col_height[7], col_height[8], col_height[9])
                if height > 15:
                    done = True
                else:
                    next_board[height][6] = 1
                    next_board[height][7] = 1
                    next_board[height][8] = 1
                    next_board[height][9] = 1
            else:
                height = max(col_height[col - 2], col_height[col - 1], col_height[col], col_height[col + 1])
                if height > 15:
                    done = True
                else:
                    next_board[height][col - 2] = 1
                    next_board[height][col - 1] = 1
                    next_board[height][col] = 1
                    next_board[height][col + 1] = 1
        elif rot == ROT_90 or rot == ROT_270:
            height = col_height[col]
            if height + 3 > 15:
                done = True
            else:
                next_board[height][col] = 1
                next_board[height + 1][col] = 1
                next_board[height + 2][col] = 1
                next_board[height + 3][col] = 1
    return next_board, done


def _get_col_heights(board):
    col_heights = np.max(board, axis=0) * (16 - np.argmax(np.flipud(board), axis=0))
    return col_heights.astype(np.int)


def _determine_action(board, current_block, next_block, weights):
    col_heights = _get_col_heights(board)
    score_list = 40 * [0]
    for i in range(40):
        col, rotate = divmod(i, 4)
        if current_block == J_BLOCK or current_block == L_BLOCK or current_block == T_BLOCK:
            if (rotate == ROT_0 or rotate == ROT_180) and (col == 1 or col == 9):
                score_list[i] = score_list[i - 4]
            elif (rotate == ROT_90 and col == 1) or (rotate == ROT_270 and col == 9):
               score_list[i] = score_list[i - 4]
            else:
                next_board, done = _add_block(board, col_heights, current_block, col, rotate)
                if done:
                    score_list[i] = -999
                else:
                    current_score = _get_score(board, next_board, weights)
                    next_board = _remove_rows(next_board)
                    next_score = _get_next_score(next_board, next_block, weights)
                    score_list[i] = current_score + next_score
        elif current_block == Z_BLOCK or current_block == S_BLOCK:
            if rotate == ROT_180 or rotate == ROT_270:
                score_list[i] = score_list[i - 2]
            elif rotate == ROT_0 and (col == 1 or col == 9):
                score_list[i] = score_list[i - 4]
            elif rotate == ROT_90 and col == 9:
                score_list[i] = score_list[i - 4]
            else:
                next_board, done = _add_block(board, col_heights, current_block, col, rotate)
                if done:
                    score_list[i] = -999
                else:
                    current_score = _get_score(board, next_board, weights)
                    next_board = _remove_rows(next_board)
                    next_score = _get_next_score(next_board, next_block, weights)
                    score_list[i] = current_score + next_score
        elif current_block == O_BLOCK:
            if rotate != ROT_0:
                score_list[i] = score_list[i - rotate]
            elif col == 1:
                score_list[i] = score_list[i - 4]
            else:
                next_board, done = _add_block(board, col_heights, current_block, col, rotate)
                if done:
                    score_list[i] = -999
                else:
                    current_score = _get_score(board, next_board, weights)
                    next_board = _remove_rows(next_board)
                    next_score = _get_next_score(next_board, next_block, weights)
                    score_list[i] = current_score + next_score
        elif current_block == I_BLOCK:
            if rotate == ROT_180 or rotate == ROT_270:
                score_list[i] = score_list[i - 2]
            elif rotate == ROT_0 and (col == 1 or col == 2):
                score_list[i] = score_list[i - 4 * col - rotate]
            elif rotate == ROT_0 and col == 9:
                score_list[i] = score_list[32]
            else:
                next_board, done = _add_block(board, col_heights, current_block, col, rotate)
                if done:
                    score_list[i] = -999
                else:
                    current_score = _get_score(board, next_board, weights)
                    next_board = _remove_rows(next_board)
                    next_score = _get_next_score(next_board, next_block, weights)
                    score_list[i] = current_score + next_score
    return np.argmax(score_list)


def _get_next_score(board, next_block, weights):
    col_heights = _get_col_heights(board)
    score_list = 40 * [0]
    for i in range(40):
        col, rotate = divmod(i, 4)
        if next_block == J_BLOCK or next_block == L_BLOCK or next_block == T_BLOCK:
            if (rotate == ROT_0 or rotate == ROT_180) and (col == 1 or col == 9):
                score_list[i] = score_list[i - 4]
            elif (rotate == ROT_90 and col == 1) or (rotate == ROT_270 and col == 9):
                score_list[i] = score_list[i - 4]
            else:
                next_board, done = _add_block(board, col_heights, next_block, col, rotate)
                if done:
                    score_list[i] = -999
                else:
                    score_list[i] = _get_score(board, next_board, weights)
        elif next_block == Z_BLOCK or next_block == S_BLOCK:
            if rotate == ROT_180 or rotate == ROT_270:
                score_list[i] = score_list[i - 2]
            elif rotate == ROT_0 and (col == 1 or col == 9):
                score_list[i] = score_list[i - 4]
            elif rotate == ROT_90 and col == 9:
                score_list[i] = score_list[i - 4]
            else:
                next_board, done = _add_block(board, col_heights, next_block, col, rotate)
                if done:
                    score_list[i] = -999
                else:
                    score_list[i] = _get_score(board, next_board, weights)
        elif next_block == O_BLOCK:
            if rotate != ROT_0:
                score_list[i] = score_list[i - rotate]
            elif col == 1:
                score_list[i] = score_list[i - 4]
            else:
                next_board, done = _add_block(board, col_heights, next_block, col, rotate)
                if done:
                    score_list[i] = -999
                else:
                    score_list[i] = _get_score(board, next_board, weights)
        elif next_block == I_BLOCK:
            if rotate == ROT_180 or rotate == ROT_270:
                score_list[i] = score_list[i - 2]
            elif rotate == ROT_0 and (col == 1 or col == 2):
                score_list[i] = score_list[i - 4 * col - rotate]
            elif rotate == ROT_0 and col == 9:
                score_list[i] = score_list[32]
            else:
                next_board, done = _add_block(board, col_heights, next_block, col, rotate)
                if done:
                    score_list[i] = -999
                else:
                    score_list[i] = _get_score(board, next_board, weights)
        return np.max(score_list)


def _get_score(board, next_board, weights):
    #if weights is None:
    #    weights = [-2.404, -1.977, -1.308, -1.263, -1.049, -0.922, 0.66, -0.161]
    isolated_block = next_board - board
    col_heights = _get_col_heights(next_board)
    holes = _get_holes(next_board, col_heights)
    wells = _get_wells(next_board)
    score = 0
    score += weights[0] * _get_rows_with_holes(holes)
    score += weights[1] * _get_column_transitions(next_board, col_heights)
    score += weights[2] * _get_num_holes(holes)
    score += weights[3] * _get_landing_height(isolated_block)
    score += weights[4] * _get_cumulative_wells(wells)
    score += weights[5] * _get_row_transitions(next_board)
    score += weights[6] * _get_eroded_cells(next_board, isolated_block)
    score += weights[7] * _get_hole_depth(next_board, holes)
    return score


def _get_holes(board, col_heights):
    holes = np.zeros((16, 10))
    for i in range(10):
        for j in range(col_heights[i]):
            if board[j][i] == 0:
                holes[j][i] = 1
    return holes


def _get_wells(board):
    wells = np.zeros((16, 10))
    for i in range(0, 16):
        for j in range(0, 10):
            if board[i][j] == 0:
                if (j == 0 and board[i][1] == 1):
                    wells[i][j] = 1
                elif j == 9 and board[i][8] == 1:
                    wells[i][j] = 1
                elif board[i][j - 1] == 1 and board[i][j + 1] == 1:
                    wells[i][j] = 1
    return wells


def _get_rows_with_holes(holes):
    return np.sum(np.max(holes, axis=1))


def _get_num_holes(holes):
    return np.sum(holes)


def _get_column_transitions(board, col_heights):
    num_trans = 10
    for i in range(10):
        current_block = 1
        for j in range(col_heights[i]):
            if board[j][i] != current_block:
                num_trans += 1
                current_block = (current_block + 1) % 2
    return num_trans


def _get_landing_height(isolated_block):
    blocks = np.max(isolated_block, axis=1)
    low = np.argmax(blocks) + 1
    high = 16 - np.argmax(np.flip(blocks))
    return (low + high) / 2


def _get_cumulative_wells(wells):
    sum_wells = np.sum(wells, axis=0)
    cum_wells = sum_wells * (sum_wells + 1) / 2
    return np.sum(cum_wells)


def _get_row_transitions(board):
    num_rows = int(np.sum(np.max(board, axis=1)))
    num_trans = 0
    for i in range(num_rows):
        current_block = 1
        for j in range(10):
            if board[i][j] != current_block:
                num_trans += 1
                current_block = (current_block + 1) % 2
        if board[i][9] == 0:
            num_trans += 1
    return num_trans


def _get_eroded_cells(next_board, isolated_block):
    rows = np.sum(next_board, axis=1)
    isolated_rows = np.sum(isolated_block, axis=1)
    full_rows = np.where(rows == 10, 1, 0)
    return np.sum(full_rows) * np.sum(isolated_rows * full_rows)


def _get_hole_depth(board, holes):
    num_depth = 0
    x, y = np.where(holes == 1)
    for i in range(len(x)):
        num_depth += np.sum(board[x[i]:, y[i]])
    return num_depth


def _remove_rows(board):
    rows = np.sum(board, axis=1)
    full_rows = np.where(rows == 10)
    next_board = np.delete(board, full_rows, axis=0)
    if next_board.shape[0] < 16:
        top_rows = np.zeros((16 - next_board.shape[0], 10)).astype(int)
        next_board = np.vstack((top_rows, next_board))
    return next_board


class Tetris_Player:
    def __init__(self, game_path,
                 state_path,
                 weights=None,
                 render=False,
                 max_lines=100):
        if weights is None:
            weights = [-2.404, -1.977, -1.308, -1.263, -1.049, -0.922, 0.66, -0.161]
        self.weights = weights
        self.render = render
        self.max_lines = max_lines
        self.name = "Player 1"
        self.num_frames = 0
        self.is_done = False
        self._env = retro.make(game=game_path,
                               state=state_path,
                               use_restricted_actions=True)

    def _perform_action(self, action):
        col, rotate = divmod(action, 4)
        if rotate == ROT_270:
            _ = self._press_button(B)
            frame = self._press_button(NO_ACTION)
        elif rotate == ROT_0:
            frame = self._press_button(NO_ACTION)
        else:
            _ = self._press_button(A)
            frame = self._press_button(NO_ACTION)
        if rotate == ROT_180:
            _ = self._press_button(A)
            frame = self._press_button(NO_ACTION)
        if col > 5:
            _ = self._press_button(RIGHT)
            frame = self._press_button(NO_ACTION)
        elif col < 5:
            _ = self._press_button(LEFT)
            frame = self._press_button(NO_ACTION)
        if col > 6:
            _ = self._press_button(RIGHT)
            frame = self._press_button(NO_ACTION)
        elif col < 4:
            _ = self._press_button(LEFT)
            frame = self._press_button(NO_ACTION)
        if col > 7:
            _ = self._press_button(RIGHT)
            frame = self._press_button(NO_ACTION)
        elif col < 3:
            _ = self._press_button(LEFT)
            frame = self._press_button(NO_ACTION)
        if col > 8:
            _ = self._press_button(RIGHT)
            frame = self._press_button(NO_ACTION)
        elif col < 2:
            _ = self._press_button(LEFT)
            frame = self._press_button(NO_ACTION)
        if col == 0:
            _ = self._press_button(LEFT)
            frame = self._press_button(NO_ACTION)
        return frame

    def _press_button(self, button):
        frame, _, _, info = self._env.step(button)
        if info["gameover"] != 0 or info["lines"] >= self.max_lines:
            self.is_done = True
        if self.render:
            self._env.render()
        return frame

    def play_single_game(self):
        _ = self._env.reset()
        if self.render:
            self._env.render()
        for i in range(np.random.randint(1, 257)):
            _, _, _, _ = self._env.step(NO_ACTION)
        frame = self._press_button(START)
        while not self.is_done:
            board, current_block, next_block = _format_frame(frame)
            if current_block == 0:
                frame = self._press_button(NO_ACTION)
            else:
                action = _determine_action(board, current_block, next_block, self.weights)
                frame = self._perform_action(action)

    def play_multiple_games(self, num_games=5):
        scores = []
        for i in range(num_games):
            _ = self._env.reset()
            if self.render:
                self._env.render()
            for i in range(np.random.randint(1, 257)):
                _, _, _, _ = self._env.step(NO_ACTION)
            frame = self._press_button(START)
            self.is_done = False
            while not self.is_done:
                board, current_block, next_block = _format_frame(frame)
                if current_block == 0:
                    frame = self._press_button(NO_ACTION)
                else:
                    action = _determine_action(board, current_block, next_block, self.weights)
                    frame = self._perform_action(action)
            _, _, _, info = self._env.step(NO_ACTION)
            scores.append(info["score"])
        return np.average(scores)

    def change_weights(self, weights):
        self.weights = weights