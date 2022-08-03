import numpy as np
from numba import njit
import src.project1.submit.AI as ai


@njit
def get_PVT_and_max(Vars):  # 10个参数
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
def value_of_positions(chessboard, color, PVT, PVT_max):
    # 假设我方是黑方，颜色为-1， 我不愿意拿到任何棋子（反向黑白棋）
    # (chessboard * PVT).sum() 是 对方拿到的子-我拿到的棋子 , 因为对方颜色是1。 我希望这个值越大越好
    # 但是不巧我是-1色，所以我要乘个-1，因为贪心函数也认为value_of_positions是对color而言越大约好的
    return ai.min_max_normalized_value(-PVT_max, PVT_max, ai.COLOR_BLACK * color * ((chessboard * PVT).sum()))


# class Agent(ai.AI):
#     def __init__(self, chessboard_size, color, time_out, Vars):
#         super().__init__(chessboard_size, color, time_out)
#         global PVT, PVT_max
#         PVT = get_PVT(Vars)
#         PVT_max = abs(PVT).sum()
#         super().set_search_depth(6)
#         super().set_evaluator(value_of_positions)

import random


def neighbours(Vars, delta, lb, ub, times=8):
    for i in random.sample(range(len(Vars)), times):
        for j in [-delta, +delta]:
            new_vars = Vars.copy()
            mask = np.zeros(len(Vars))
            mask[i] = 1
            new_vars += j * mask
            if (lb <= new_vars).all() and (new_vars<= ub).all():
                yield new_vars


@njit
def neighbours_continuous(Vars: np.ndarray, lb=-1, ub=1, scale=0.1, times=100):
    res = [Vars]
    for i in range(times):
        noise = np.random.normal(loc=0, scale=scale, size=(len(Vars)))
        res.append(np.clip(Vars + noise, lb, ub))
    return res


if __name__ == '__main__':
    dim = 16
    # varTypes = np.ones(dim) # 整数
    varTypes = np.zeros(dim)  # 实数
    lb = np.ones(dim) * -20
    ub = np.ones(dim) * 20
    lb[10:] = 0
    ub[10:] = 1
    delta = np.ones(dim)
    delta[10:] = 0.1
    Vars = np.array([3, 19, 10, 5,18, 10, 10,10, 8,3, 0.7, 0.1, 0.2,0.6, 0.2, 0.2])
    print(list(neighbours(Vars, delta, lb, ub, 4)))
    print(list(neighbours(Vars, delta, lb, ub, 8)))


    # Vars = np.array([1, 8, 3, 7, 3, 2, 5, 6, 6, 4])
    # PVT, PVT_max = get_PVT_and_max(Vars)
    # print(value_of_positions(np.ones((8, 8)), 1, PVT, PVT_max))  # 我是白方，全是白子
    # print(value_of_positions(-np.ones((8, 8)), 1, PVT, PVT_max))  # 我是白方，全是黑子
    # Vars = np.ones(10)
    # PVT, PVT_max = get_PVT_and_max(Vars)
    # print(value_of_positions(PVT, 1, PVT, PVT_max))  # 我是白方，全是白子
    # print(value_of_positions(PVT, -1, PVT, PVT_max))  # 我是黑方，全是白子
