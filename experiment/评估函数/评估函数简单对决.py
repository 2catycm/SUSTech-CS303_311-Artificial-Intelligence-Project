from simulator import Simulator
from greedy_ai import GreedyAI
import src.project1.submit.AI as ai
from numba import njit

import numpy as np

chessboard_size = 8
time_out = 5


# 放到0-1之间
def min_max_normalized_function(lower_bound, upper_bound, fun):
    return lambda a, p: (fun(a, p) - lower_bound) / (upper_bound - lower_bound)


# 放到0-1之间
def min_max_normalized_value(lower_bound, upper_bound, value):
    return (value - lower_bound) / (upper_bound - lower_bound)


# action greedy
def middle_action_first(a, p):
    return -np.linalg.norm(np.array(p) - np.array([a.chessboard_size / 2, a.chessboard_size / 2]))


def eat_less_first(a, p):
    old_black, old_white = a.reversi_env.piece_cnt(a.chessboard)
    new_chessboard = a.reversi_env.updated_chessboard(a.chessboard, a.color, p)
    new_black, new_white = a.reversi_env.piece_cnt(new_chessboard)
    eat = abs((new_black - new_white) - (old_black - old_white) + a.color * 1)
    return -eat


# state greedy
def become_less_first(a, p):
    new_chessboard = a.reversi_env.updated_chessboard(a.chessboard, a.color, p)
    new_black, new_white = a.reversi_env.piece_cnt(new_chessboard)
    return a.color * (new_black - new_white)


alpha_beta_goes_to_the_end = 0  # 这是个稀罕情况


# 我们把alpha beta 搜索视作是一种特殊的价值函数，给定 棋盘和颜色 得到的一个高阶评估函数。
# 求解需要递归求解而已。
# 编程上，我们为了符合前面的action_Greedy接口，求的是每一个state的min_value
def alpha_beta_search(a, p, depth=3):
    my_color = a.color
    env = a.reversi_env
    max_piece_cnt = a.chessboard_size ** 2

    # @njit(cache=True)
    def max_value(state, color, alpha, beta, remaining_depth=depth):
        global alpha_beta_goes_to_the_end
        if env.is_terminal(state, color):
            alpha_beta_goes_to_the_end += 1
            diff = my_color * env.get_winning_piece_cnt(state)
            return min_max_normalized_value(-max_piece_cnt, max_piece_cnt, diff) * 2

        value = -np.inf
        actions = sorted(env.actions(state, color, numpy=True), key=lambda x: middle_action_first(a, x),
                         reverse=True)  # 先遍历最优的
        if len(actions) == 0:
            # 只能选择跳过这个action，value为对方的value
            return min_value(state, -color, alpha, beta, remaining_depth - 1)

        if remaining_depth <= 1:
            # 不需要为middle_action_first 构建新的棋盘。
            return min_max_normalized_value(2 ** 0.5 / 2, 2 ** 0.5 / 2 * a.chessboard_size,
                                            middle_action_first(a, actions[0]))

        for action in actions:
            new_chessboard = env.updated_chessboard(state, color, action)
            new_value = min_value(new_chessboard, -color, alpha, beta, remaining_depth - 1)
            if new_value > value:
                value = new_value
                alpha = max(alpha, value)
            if value >= beta:  # 这是？剪枝
                return value
        return value

    # @njit(cache=True)
    def min_value(state, color, alpha, beta, remaining_depth=depth):
        global alpha_beta_goes_to_the_end
        if env.is_terminal(state, color):
            alpha_beta_goes_to_the_end += 1
            diff = my_color * env.get_winning_piece_cnt(state)
            return min_max_normalized_value(-max_piece_cnt, max_piece_cnt, diff) * 2

        value = +np.inf
        actions = sorted(env.actions(state, color, numpy=True), key=lambda x: middle_action_first(a, x),
                         reverse=True)  # 还是要reverse，因为我们现在的评估函数是action评估，最大值才对对手有利
        if len(actions) == 0:
            # 只能选择跳过这个action，value为对方的value
            return max_value(state, -color, alpha, beta, remaining_depth - 1)

        if remaining_depth <= 1:
            # 不需要为middle_action_first 构建新的棋盘。
            return min_max_normalized_value(2 ** 0.5 / 2, 2 ** 0.5 / 2 * a.chessboard_size,
                                            middle_action_first(a, actions[0]))

        for action in actions:
            new_chessboard = env.updated_chessboard(state, color, action)
            new_value = max_value(new_chessboard, -color, alpha, beta, remaining_depth - 1)
            if new_value < value:
                value = new_value
                beta = min(beta, value)
            if value <= alpha:  # 这是？剪枝
                return value
        return value

    return min_value(env.updated_chessboard(a.chessboard, my_color, p), -my_color, -np.inf,
                     np.inf)  # 可能错过了一些剪枝，因为你是从min_value开始的


# greedy_functions = [middle_action_first, middle_action_first]
# greedy_functions = [middle_action_first, eat_less_first, become_less_first]
greedy_functions = [middle_action_first, eat_less_first, become_less_first, alpha_beta_search]
# greedy_functions = [eat_less_first, become_less_first, alpha_beta_search]

color_name = {ai.COLOR_BLACK:"Black", ai.COLOR_WHITE:"White"}
length = len(greedy_functions)
scores = [0 for i in range(length)]
for i in range(length):
    for j in range(i + 1, length):
        print(f"{greedy_functions[i].__name__} is playing with {greedy_functions[j].__name__}")
        for i_color in [ai.COLOR_BLACK, ai.COLOR_WHITE]:
            print(f"{greedy_functions[i].__name__} is {color_name[i_color]}")
            agents = {i_color: GreedyAI(chessboard_size, i_color, greedy_functions[i]),
                      -i_color: GreedyAI(chessboard_size, -i_color, greedy_functions[j])}
            simulator = Simulator(chessboard_size, time_out, agents)
            winner = simulator.quick_run()
            if winner == i_color:
                print(f"{greedy_functions[i].__name__} won.")
                scores[i] += 1
            elif winner == -i_color:
                print(f"{greedy_functions[j].__name__} won.")
                scores[j] += 1
            else:
                print("draw.")

print(scores)
print(alpha_beta_goes_to_the_end)  # depth为2 时， 8/6， 每场1.5次搜索到结尾
