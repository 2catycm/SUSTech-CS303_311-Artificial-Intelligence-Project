import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)


class ReversiEnv(object):
    udlr_luruldrd = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]])

    def __init__(self, chessboard_size):
        self.chessboard_size = chessboard_size

    def actions(self, chessboard, color, candidate_list=None, numpy=False):
        if candidate_list is None:
            candidate_list = []
        arg_where = np.argwhere(chessboard == 0).tolist()
        for index in arg_where:
            is_valid_move = self.is_valid_move(chessboard, color, index)
            if is_valid_move:
                candidate_list.append(tuple(index) if not numpy else index)
        return candidate_list

    # 注意可能多个方向上都会发生变化
    def updated_chessboard(self, chessboard, color, index: np.ndarray) -> np.ndarray:
        new_chessboard = chessboard.copy()
        for direction in ReversiEnv.udlr_luruldrd:
            neighbour = index + direction
            if (not self.is_valid_index(neighbour)) or chessboard[tuple(neighbour)] != -color:
                continue
            while self.is_valid_index(neighbour) and chessboard[tuple(neighbour)] == -color:
                neighbour += direction
            if self.is_valid_index(neighbour) and chessboard[tuple(neighbour)] == color:  # 找到友军了
                while (neighbour != (index-direction)).any():
                    new_chessboard[tuple(neighbour)] = color  # 修改棋盘
                    neighbour -= direction
                break
        return new_chessboard

    def is_terminal(self, chessboard, color) -> bool:
        return len(self.actions(chessboard, color)) == 0 and len(self.actions(chessboard, -color)) == 0

    def get_winner(self, chessboard):
        diff = self.get_winning_piece_cnt(chessboard)
        return COLOR_BLACK if diff < 0 else COLOR_WHITE if diff > 0 else COLOR_NONE

    # 返回黑的比白的多多少片
    def get_winning_piece_cnt(self, chessboard) -> int:
        black, white = self.piece_cnt(chessboard)
        return black - white

    def is_valid_index(self, index) -> bool:
        return 0 <= index[0] < self.chessboard_size and 0 <= index[1] < self.chessboard_size

    def is_valid_move(self, chessboard, color, index: np.ndarray) -> bool:
        # 经典代码，暂时不要重构
        if chessboard[tuple(index)] != 0:
            return False
        for i, neighbour in enumerate(index + ReversiEnv.udlr_luruldrd):
            if (not self.is_valid_index(neighbour)) or chessboard[tuple(neighbour)] != -color:
                continue
            while self.is_valid_index(neighbour):
                if chessboard[tuple(neighbour)] == color:  # 找到友军了
                    return True
                elif chessboard[tuple(neighbour)] != -color:
                    break  # 如果提前出现了空格也不对
                neighbour += ReversiEnv.udlr_luruldrd[i]
        return False

    def piece_cnt(self, chessboard):
        return np.count_nonzero(chessboard == COLOR_BLACK), np.count_nonzero(chessboard == COLOR_WHITE)


class Evaluator:
    def __init__(self, chessboard_size):
        self.chessboard_size = chessboard_size

    # 评估函数
    # def value_evaluation(self, chessboard):
    #     valid = self.find_valid(chessboard, numpy=True)
    #     return sum(map(self.single_piece_evaluation, valid))

    def single_piece_evaluation(self, piece: np.array):
        # 中间的价值大
        return -np.linalg.norm(np.array(piece) - np.array([self.chessboard_size / 2, self.chessboard_size / 2]))

    # def decision_evaluation(self, piece: np.array, chessboard):
    #     new_chessboard = self.decision_evolve(tuple(piece), chessboard)
    #     return self.value_evaluation(new_chessboard)

    def decision_evolve(self, piece: tuple, chessboard):
        new_chessboard = chessboard.copy()
        new_chessboard[piece] = self.color
        return new_chessboard


# don't change the class name
class AI(object):

    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size  # 是个int， 棋盘是 chessboard_size x chessboard_size
        self.color = color  # 黑或白
        self.time_out = time_out  # 单位s
        self.candidate_list = []
        self.reversi_env = ReversiEnv(chessboard_size)
        self.evaluator = self.reversi_env.evaluator

    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        self.candidate_list = self.reversi_env.actions(chessboard, self.color, self.candidate_list)
        if len(self.candidate_list) == 0:
            return
        else:
            # decision = self.candidate_list[random.randint(0, len(self.candidate_list)-1)]
            decision = max(self.candidate_list, key=self.evaluator.single_piece_evaluation)
            self.execute_decision(decision)

    # 根据 decision:2维元组 修改 candidate_list
    def execute_decision(self, decision):
        self.candidate_list.append(decision)
