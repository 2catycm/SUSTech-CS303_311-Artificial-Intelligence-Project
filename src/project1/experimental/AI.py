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
def is_terminal(chessboard, color=COLOR_BLACK) -> bool:
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


@njit(inline='always')
def min_max_normalized_value(lower_bound, upper_bound, value):
    return (value - lower_bound) / (upper_bound - lower_bound)


Vars = [9, 10, 18, 5, 19, 15, 18, 13, 8, 3]


@njit
def get_PVT_and_max(Vars:np.ndarray):  # 10个参数
    corn, c, cc, ccc, x, xc, xcc, xx, xxc, xxx = Vars
    PVT = np.array(
        [[corn, c, cc, ccc, ccc, cc, c, corn],
         [c, x, xc, xcc, xcc, xc, x, c],
         [cc, xc, xx, xxc, xxc, xx, xc, cc],
         [ccc, xcc, xxc, xxx, xxx, xxc, xcc, ccc],
         [ccc, xcc, xxc, xxx, xxx, xxc, xcc, ccc],
         [cc, xc, xx, xxc, xxc, xx, xc, cc],
         [c, x, xc, xcc, xcc, xc, x, c],
         [corn, c, cc, ccc, ccc, cc, c, corn]])
    PVT_max = np.absolute(PVT).sum()
    return PVT, PVT_max


PVT, PVT_max = get_PVT_and_max(np.array(Vars))


@njit
def value_of_positions(chessboard, color):
    # 假设我方是黑方，颜色为-1， 我不愿意拿到任何棋子（反向黑白棋）
    # (chessboard * PVT).sum() 是 对方拿到的子-我拿到的棋子 , 因为对方颜色是1。 我希望这个值越大越好
    # 但是不巧我是-1色，所以我要乘个-1，因为贪心函数也认为value_of_positions是对color而言越大约好的
    return min_max_normalized_value(-PVT_max, PVT_max, COLOR_BLACK * color * ((chessboard * PVT).sum()))


@njit
def hash_board(chessboard):
    return hash(str(chessboard))


@njit
def insertion_sort(acts, new_chessboards, current_color):
    """
    对 new_chessboards 按照简单评估函数，根据当前颜色的胜率进行从大到小排序。
    :param acts: 相应的action也要一起排序. 要求是list, 里面是np.array
    :param current_color: 当前的阵营。
    :param new_chessboards: 要排序的棋盘布局
    """
    new_chessboards_values = [value_of_positions(c, current_color) for c in new_chessboards]
    for i in range(1, len(new_chessboards)):
        pre_index = i - 1
        current_c, current_a = new_chessboards[i], acts[i]
        current_value = new_chessboards_values[i]
        while pre_index >= 0 and new_chessboards_values[pre_index] < current_value:
            new_chessboards[pre_index + 1] = new_chessboards[pre_index]
            acts[pre_index + 1] = acts[pre_index]
            pre_index -= 1
        new_chessboards[pre_index + 1] = current_c
        acts[pre_index + 1] = current_a


# don't change the class name
class AI(object):
    def __init__(self, local_chessboard_size, color, time_out):
        global chessboard_size
        chessboard_size = self.chessboard_size = local_chessboard_size
        self.color = color  # 黑或白
        self.time_out = time_out * 0.98  # 防止时间波动导致超时
        self.start_time = None
        self.candidate_list = []
        self.rounds = 4 if color == COLOR_BLACK else 5

    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        numpy_list = actions(chessboard, self.color)
        self.candidate_list = list(map(index2, numpy_list))
        if len(self.candidate_list) == 0 or len(self.candidate_list) == 1:
            return
        self.start_time = time.time()
        value, decision = alpha_beta_search(chessboard, self.color)
        self.execute_decision(decision)
        self.rounds += 1

    def execute_decision(self, decision):
        self.candidate_list.append(index2(decision))


def iterative_deepening_search(start_time, time_out, memory_out):
    current_depth = 2
    hash_table = {}


# @njit
def alpha_beta_search(chessboard, current_color, remaining_depth=100, alphas=np.array([-np.inf, -np.inf]),
                      hash_table=None):
    """

    :param chessboard:
    :param current_color:
    :param remaining_depth: 0 表示 直接对节点估值，不合法。 1表示一层贪心。 根据时间资源和回合数，请合理分配搜索深度。目前知道10层全回合OK的
    :param alphas: 0:到目前为止，路径上发现的 color=-1这个agent 的最佳选择值
                  1:到目前为止，路径上发现的 color= 1这个agent 的最佳选择值
    :param hash_table:
    :return: 返回对于chessboard,color这个节点，它最大的选择值是多少，以及它选择了哪个子节点。
    """
    alphas = alphas.copy()  # 防止修改上面的alphas
    if is_terminal(chessboard):
        utility = current_color * get_winner(chessboard)  # winner的颜色和我相等，就是1（颜色的平方性质）， 和我的颜色不等，就是-1.
        return min_max_normalized_value(-1, 1, utility), None  # 满足截断性。由于其他价值函数也归一化了，0和1就是最小值和最大值

    acts = actions(chessboard, current_color)
    if len(acts) == 0:
        # 只能选择跳过这个action，value为对方的value
        value, move = alpha_beta_search(chessboard, -current_color, remaining_depth - 1, alphas)
        return -value, None  # 对手的值是和我反的。 我方没有action可以做。
    new_chessboards = [updated_chessboard(chessboard, current_color, a) for a in acts]  # 用最多10倍内存换一半时间（排序和实际操作共用结果）
    insertion_sort(acts, new_chessboards, current_color)

    if remaining_depth <= 1:  # 比如要求搜索1层，就是直接对max节点的所有邻接节点排序返回最大的。
        return value_of_positions(new_chessboards[0], current_color), acts[0]  # 评价永远是根据我方的棋盘

    value, move = -np.inf, None  # 写在一起。每个节点都尝试让自己的价值最大化
    this_color_idx, other_color_idx = int((current_color + 1) // 2), int((-current_color + 1) // 2)
    for i, new_chessboard in enumerate(new_chessboards):
        action = acts[i]

        new_value, t = alpha_beta_search(new_chessboard, -current_color, remaining_depth - 1, alphas)
        new_value = -new_value

        if new_value > value:
            value, move = new_value, action

            alphas[this_color_idx] = max(alphas[this_color_idx], value)
        if value >= alphas[other_color_idx]:  # 这是beta剪枝, 因为min的某个祖先的要求，这个max被其父min剪掉。
            return value, move
    return value, move
