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
        self.find_valid(chessboard)
        if len(self.candidate_list)==0:
            self.candidate_list = []
        else:
            self.execute_decision(self.candidate_list[random.randint(0, len(self.candidate_list)-1)])
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
    def is_valid_index(self, index):
        return 0<=index[0]<self.chessboard_size and 0<=index[1]<self.chessboard_size
    def is_valid_move(self, index: np.ndarray, chessboard):
        if chessboard[tuple(index)] != 0:
            return False
        for i, neighbour in enumerate(index + AI.udlr_luruldrd):
            if (not self.is_valid_index(neighbour)) or chessboard[tuple(neighbour)]!=-self.color:
                continue
            while self.is_valid_index(neighbour):
                if chessboard[tuple(neighbour)]==self.color:
                    # 找到友军了
                    return True
                elif chessboard[tuple(neighbour)]!=-self.color:
                    break  # 如果提前出现了空格也不对
                neighbour += AI.udlr_luruldrd[i]
        return False
    # void, 修改决策之前的 candidate_list
    def find_valid(self, chessboard):
        arg_where = np.argwhere(chessboard == 0).tolist()
        # self.candidate_list = np.argwhere(is_valid_move(arg_where))
        for index in arg_where:
            if self.is_valid_move(index, chessboard):
                self.candidate_list.append(tuple(index))
    # 根据 decision:2维元组 修改 candidate_list
    def execute_decision(self, decision):
        self.candidate_list.append(decision)
