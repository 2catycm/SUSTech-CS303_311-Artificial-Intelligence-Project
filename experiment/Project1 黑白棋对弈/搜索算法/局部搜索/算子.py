import numpy as np
from numba import njit
import src.project1.submit.AI as ai
import random

chessboard_size = 8
COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
max_piece_cnt = chessboard_size ** 2


@njit(inline='always')
def index2(x):
    return x[0], x[1]


@njit(inline='always')
def symmetry_normalized_value(lower_bound, upper_bound, value):
    # 放缩到 [-1, 1] 范围，好处是满足 对手对称性
    return 2 * (value - lower_bound) / (upper_bound - lower_bound) - 1


@njit
def get_PVT_and_max(Vars: np.ndarray):  # 10个参数
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


@njit
def random_base_line(chessboard, color, rounds):
    return random.random()


@njit
def value_of_positions(chessboard, color, PVT, PVT_max):
    # 假设我方是黑方，颜色为-1， 我不愿意拿到任何棋子（反向黑白棋）
    # (chessboard * PVT).sum() 是 对方拿到的子-我拿到的棋子 , 因为对方颜色是1。 我希望这个值越大越好
    # 但是不巧我是-1色，所以我要乘个-1，因为贪心函数也认为value_of_positions是对color而言越大约好的
    return symmetry_normalized_value(-PVT_max, PVT_max, COLOR_BLACK * color * ((chessboard * PVT).sum()))


@njit
def value_of_mobility(chessboard, color, PVT, PVT_max):
    my_acts = ai.actions(chessboard, color)
    opponent_acts = ai.actions(chessboard, -color)
    value = 0
    for act in my_acts:
        value += PVT[index2(act)]
    for act in opponent_acts:
        value -= PVT[index2(act)]
    return symmetry_normalized_value(-PVT_max / max_piece_cnt * 12, PVT_max / max_piece_cnt * 12, value)


@njit
def value_of_edge_stability(chessboard, color):
    value = 0
    corners_and_directions = np.array([[[0, 0], [1, 0], [0, 1]],
                                       [[0, chessboard_size - 1], [1, 0], [0, -1]],
                                       [[chessboard_size - 1, 0], [-1, 0], [0, 1]],
                                       [[chessboard_size - 1, chessboard_size - 1], [-1, 0], [0, -1]]
                                       ])
    for corner, ud, lr in corners_and_directions:
        corner_color = chessboard[index2(corner)]
        if corner_color == COLOR_NONE:
            continue
        value += corner_color  # 角本身算一个子
        for d in [ud, lr]:
            current = corner + d
            cnt = 1
            while cnt <= chessboard_size - 1 and chessboard[index2(current)] == corner_color:
                value += corner_color  # 被两个角夹住的话当且仅当整个边都被占领，此时确实认为会劣势一些
                current = current + d
                cnt += 1
            if cnt == chessboard_size:
                if corner[0] - corner[1] != 0:
                    value -= (chessboard_size - 1) * corner_color
                else:
                    value -= corner_color  # 角不算

    return symmetry_normalized_value(-28, 28, color * COLOR_BLACK * value)  # 如果全是白方的稳定子，算出来28，对黑方有利，几乎必胜。


if __name__ == '__main__':
    c = np.ones((chessboard_size, chessboard_size))
    print(value_of_edge_stability(c, COLOR_BLACK))
    PVT, PVT_sum = get_PVT_and_max(np.array([9, 10, 18, 5, 19, 15, 18, 13, 8, 3]))
    chessboard = np.array(
        [[-1, 1, 1, 1, 1, 1, 1, -1, ],
         [-1, 1, 1, 1, 1, 1, 1, -1, ],
         [-1, 1, -1, 1, -1, 1, 1, -1, ],
         [-1, 1, -1, 1, 1, -1, 1, -1, ],
         [-1, 1, 1, -1, -1, -1, 1, -1, ],
         [-1, -1, -1, -1, -1, -1, 1, -1, ],
         [0, 0, -1, -1, -1, 1, 1, -1, ],
         [0, -1, -1, -1, -1, -1, 1, -1, ]])
    print(value_of_edge_stability(chessboard, COLOR_BLACK))
    print(value_of_mobility(chessboard, COLOR_BLACK, PVT, PVT_sum))
