import numpy as np
from numba import njit
from numba import prange
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
random.seed(0)
chessboard_size = 8  # 是个int， 棋盘是 chessboard_size x chessboard_size
max_piece_cnt = chessboard_size ** 2

######################     Reversi Env   ######################
udlr_luruldrd = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]])


@njit(inline='always')
def index2(x):
    return x[0], x[1]


@njit()
def actions(chessboard, color):
    """
    根据当前棋盘状况和当前准备落子的颜色，得到所给颜色的所有遵守规则的行动。
    :param chessboard: 棋盘
    :param color: 颜色
    :return:
    """
    candidate_list = []
    arg_where = np.argwhere((chessboard == 0))
    for index in arg_where:
        valid = is_valid_move(chessboard, color, index)
        if valid:
            candidate_list.append(index)
    return candidate_list


# 注意可能多个方向上都会发生变化
@njit()
def updated_chessboard(chessboard, color, index: np.ndarray) -> np.ndarray:
    new_chessboard = chessboard.copy()
    for direction in udlr_luruldrd:
        neighbour = index + direction
        if (not is_valid_index(neighbour)) or chessboard[index2(neighbour)] != -color:
            continue
        while is_valid_index(neighbour) and chessboard[index2(neighbour)] == -color:
            neighbour += direction
        if is_valid_index(neighbour) and chessboard[index2(neighbour)] == color:  # 找到友军了
            while (neighbour != (index - direction)).any():
                new_chessboard[index2(neighbour)] = color  # 修改棋盘
                neighbour -= direction
            break
    return new_chessboard


@njit(inline='always')
def is_terminal(chessboard, color) -> bool:
    return len(actions(chessboard, color)) == 0 and len(actions(chessboard, -color)) == 0


@njit(inline='always')
def get_winner(chessboard):
    diff = get_winning_piece_cnt(chessboard)
    return COLOR_BLACK if diff < 0 else COLOR_WHITE if diff > 0 else COLOR_NONE


@njit(inline='always')
def get_winning_piece_cnt(chessboard) -> int:
    # 返回黑的比白的多多少片
    black, white = piece_cnt(chessboard)
    return black - white


@njit(inline='always')
def is_valid_index(index) -> bool:
    return 0 <= index[0] < chessboard_size and 0 <= index[1] < chessboard_size


@njit()
def is_valid_move(chessboard, color, index: np.ndarray) -> bool:
    # 经典代码，暂时不要重构
    if chessboard[index2(index)] != 0:
        return False
    for i, neighbour in enumerate(index + udlr_luruldrd):
        if (not is_valid_index(neighbour)) or chessboard[index2(neighbour)] != -color:
            continue
        while is_valid_index(neighbour):
            if chessboard[index2(neighbour)] == color:  # 找到友军了
                return True
            elif chessboard[index2(neighbour)] != -color:
                break  # 如果提前出现了空格也不对
            neighbour += udlr_luruldrd[i]
    return False


@njit(inline='always')
def piece_cnt(chessboard):
    return np.count_nonzero(chessboard == COLOR_BLACK), np.count_nonzero(chessboard == COLOR_WHITE)


######################  Evaluator ######################

def middle_action_first(piece: np.array):
    # 中间的价值大
    return -np.linalg.norm(np.array(piece) - np.array([chessboard_size / 2, chessboard_size / 2]))


PVT = np.array([[500, -25, 10, 5, 5, 10, -25, 500],
                [-25, -45, 1, 1, 1, 1, -45, -25],
                [10, 1, 3, 2, 2, 3, 1, 10],
                [5, 1, 2, 1, 1, 2, 1, 5],
                [5, 1, 2, 1, 1, 2, 1, 5],
                [10, 1, 3, 2, 2, 3, 1, 10],
                [-25, -45, 1, 1, 1, 1, -45, -25],
                [500, -25, 10, 5, 5, 10, -25, 500]])  # position value table

# PVT = np.ones((chessboard_size, chessboard_size))
PVT_max = abs(PVT).sum()


@njit(inline='always')
def min_max_normalized_value(lower_bound, upper_bound, value):
    return (value - lower_bound) / (upper_bound - lower_bound)


@njit(inline='always')
def value_of_position(chessboard, color):
    return (chessboard * PVT).sum() * color
    # return -(chessboard * PVT[2:6, 2:6]).sum() * color


# don't change the class name
class AI(object):
    def __init__(self, local_chessboard_size, color, time_out, evaluator=value_of_position):
        global chessboard_size
        chessboard_size = self.chessboard_size = local_chessboard_size
        self.color = color  # 黑或白
        self.time_out = time_out*0.6  # 单位s
        self.start_time = None
        self.candidate_list = []
        self.rounds = 4 if color == COLOR_BLACK else 5
        self.evaluator = evaluator

    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        numpy_list = actions(chessboard, self.color)
        self.candidate_list = list(map(index2, numpy_list))
        if len(self.candidate_list) == 0 or len(self.candidate_list) == 1:
            return
        else:
            value, decision = self.alpha_beta_search(chessboard, self.color, self.evaluator)
            self.execute_decision(decision)

    def execute_decision_idx(self, decision_idx):
        # 根据 decision:2维元组 修改 candidate_list
        self.candidate_list.append(self.candidate_list[decision_idx])

    def execute_decision(self, decision):
        self.candidate_list.append(index2(decision))

    @staticmethod
    def alpha_beta_search(chessboard, my_color, evaluator=value_of_position):
        def max_value(state, color, alpha, beta, evaluator=value_of_position):
            if is_terminal(state, color):
                diff = my_color * COLOR_WHITE * get_winning_piece_cnt(state)  # 比如我是黑方，黑的比白的多10片，属于劣势，所以my_color=-1
                # return min_max_normalized_value(-max_piece_cnt, max_piece_cnt, diff) * 2, None
                return diff, None

            value, move = -np.inf, None
            acts = actions(state, color)
            if len(acts) == 0:
                # 只能选择跳过这个action，value为对方的value
                return min_value(state, -color, alpha, beta, evaluator)
            # acts.sort(key=lambda a: value_of_position(updated_chessboard(state, -1, a), -1),
            acts.sort(key=lambda a: evaluator(updated_chessboard(state, color, a), my_color),
                      reverse=True)  # 先遍历我下了棋后，对于我方而言最优的

            if remaining_depth <= 1:
                # 不需要为middle_action_first 构建新的棋盘。
                # return min_max_normalized_value(-PVT_max, +PVT_max,
                #                                 value_of_position(updated_chessboard(state, color, acts[0]), my_color)), None
                return evaluator(updated_chessboard(state, color, acts[0]), my_color), None  # 评价永远是根据我方的棋盘

            for action in acts:
                new_chessboard = updated_chessboard(state, color, action)
                new_value, t = min_value(new_chessboard, -color, alpha, beta, remaining_depth - 1)
                if new_value > value:
                    value, move = new_value, action
                    alpha = max(alpha, value)
                if value >= beta:  # 这是beta剪枝, 因为min的某个祖先的要求，这个max被其父min剪掉。
                    return value, move
            return value, move

        def min_value(state, color, alpha, beta, remaining_depth=depth, evaluator=value_of_position):
            if is_terminal(state, color):
                diff = my_color * COLOR_WHITE * get_winning_piece_cnt(state)
                # return min_max_normalized_value(-max_piece_cnt, max_piece_cnt, diff) * 2, None
                return diff, None

            value, move = +np.inf, None
            acts = actions(state, color)
            if len(acts) == 0:
                # 只能选择跳过这个action，value为对方的value
                return max_value(state, -color, alpha, beta, remaining_depth - 1)
            acts.sort(
                key=lambda a: evaluator(updated_chessboard(state, color, a), my_color),
                reverse=False)  # 先遍历我下了棋后，对于我方而言最差的。

            if remaining_depth <= 1:
                # 不需要为middle_action_first 构建新的棋盘。
                # return min_max_normalized_value(-PVT_max, +PVT_max,
                #                                 evaluator(updated_chessboard(state, color, acts[0]), my_color)),None
                return evaluator(updated_chessboard(state, color, acts[0]), my_color), None
            for action in acts:
                new_chessboard = updated_chessboard(state, color, action)
                new_value, t = max_value(new_chessboard, -color, alpha, beta, remaining_depth - 1)
                if new_value < value:
                    value, move = new_value, action
                    beta = min(beta, value)
                if value <= alpha:  # 这是alpha剪枝
                    return value, move
            return value, move

        return max_value(chessboard, my_color, -np.inf, +np.inf, depth, evaluator)
