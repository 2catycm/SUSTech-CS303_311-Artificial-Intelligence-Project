import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)


# don't change the class name
class AI(object):
    udlr_luruldrd = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]])

    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size  # 是个int， 棋盘是 chessboard_size x chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need add your decision into your candidate_list. System will get the end of your candidate_list as your
        # decision .
        self.candidate_list = []

    # The input is current chessboard.
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        # ==================================================================
        # Write your algorithm here
        # Here is the simplest sample:Random decision
        self.candidate_list = self.find_valid(self.chessboard_size, self.color, chessboard)
        if len(self.candidate_list) == 0:
            return
        else:
            self.execute_decision(self.candidate_list[random.randint(0, len(self.candidate_list) - 1)])
        # ==============Find new pos========================================
        # Make sure that the position of your decision in chess board is empty.
        # If not, the system will return error.
        # Add your decision into candidate_list, Records the chess board
        # You need add all the positions which is valid
        # candidate_list example: [(3,3),(4,4)]
        # You need append your decision at the end of the candidate_list,
        # we will pickthe last element of the candidate_list as the position you choose
        # If there is no valid position, you must return an empty list.

    # 基本环境
    @staticmethod
    def is_valid_index(chessboard_size, index):
        return 0 <= index[0] < chessboard_size and 0 <= index[1] < chessboard_size
    @staticmethod
    def is_valid_move(chessboard_size, color, index: np.ndarray, chessboard):
        if chessboard[tuple(index)] != 0:
            return False
        for i, neighbour in enumerate(index + AI.udlr_luruldrd):
            if (not AI.is_valid_index(chessboard_size, neighbour)) or chessboard[tuple(neighbour)] != -color:
                continue
            while AI.is_valid_index(chessboard_size, neighbour):
                if chessboard[tuple(neighbour)] == color:
                    # 找到友军了
                    return True
                elif chessboard[tuple(neighbour)] != -color:
                    break  # 如果提前出现了空格也不对
                neighbour += AI.udlr_luruldrd[i]
        return False

    @staticmethod
    def find_valid(chessboard_size, color, chessboard, candidate_list=None):
        if candidate_list is None:
            candidate_list = []
        arg_where = np.argwhere(chessboard == 0).tolist()
        for index in arg_where:
            if AI.is_valid_move(chessboard_size, color, index, chessboard):
                candidate_list.append(tuple(index))
        return candidate_list

    # 根据 decision:2维元组 修改 candidate_list
    def execute_decision(self, decision):
        self.candidate_list.append(decision)
