import numpy as np
from numba import njit, typed
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


@njit(inline='always')
def symmetry_normalized_value(lower_bound, upper_bound, value):
    # 放缩到 [-1, 1] 范围，好处是满足 对手对称性
    return 2 * (value - lower_bound) / (upper_bound - lower_bound) - 1


# Vars = [9, 10, 18, 5, 19, 15, 18, 13, 8, 3]
Vars = np.array([3, 19, 10, 5,
                 18, 10, 10,
                 10, 8,
                 3,
                 0.7, 0.1, 0.2,
                 0.6, 0.2, 0.2])


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


PVT, PVT_max = get_PVT_and_max(Vars[0:10])


@njit
def value_of_positions(chessboard, color):
    # 假设我方是黑方，颜色为-1， 我不愿意拿到任何棋子（反向黑白棋）
    # (chessboard * PVT).sum() 是 对方拿到的子-我拿到的棋子 , 因为对方颜色是1。 我希望这个值越大越好
    # 但是不巧我是-1色，所以我要乘个-1，因为贪心函数也认为value_of_positions是对color而言越大约好的
    return symmetry_normalized_value(-PVT_max, PVT_max, COLOR_BLACK * color * ((chessboard * PVT).sum()))


@njit
def value_of_mobility(chessboard, color):
    my_acts = actions(chessboard, color)
    opponent_acts = actions(chessboard, -color)
    value = 0
    for act in my_acts:
        value += PVT[index2(act)]  # 我的行动力， 我想下的程度
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


# @njit
# def 组合策略1(chessboard, color, rounds):
#     return value_of_positions(chessboard, color, rounds)  # 回退到旧版的方法

@njit
def 组合策略1(chessboard, color, rounds):
    v_position = value_of_positions(chessboard, color)
    v_mobility = value_of_mobility(chessboard, color)
    v_edge_sta = value_of_edge_stability(chessboard, color)
    if rounds <= 24:
        return v_position * Vars[10] + v_mobility * Vars[11] + v_edge_sta * Vars[12]
    else:
        return v_position * Vars[13] + v_mobility * Vars[14] + v_edge_sta * Vars[15]


# don't change the class name
class AI(object):
    def __init__(self, local_chessboard_size, color, time_out):
        global chessboard_size
        chessboard_size = self.chessboard_size = local_chessboard_size
        self.color = color  # 黑或白
        self.time_out = time_out * 0.95  # 防止时间波动导致超时
        self.candidate_list = []
        self.rounds = 4 if color == COLOR_BLACK else 5

    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        numpy_list = actions(chessboard, self.color)
        self.candidate_list = list(map(index2, numpy_list))
        if len(self.candidate_list) == 0 or len(self.candidate_list) == 1:  # 不要self.rounds<=5， 趁着第一回合jit一下
            print("Only one or no action. ")
            return
        value, decision = iterative_deepening_search(chessboard, self.color, self.time_out, self.rounds, self.execute_decision)
        if decision is not None:
            self.execute_decision(decision)
        self.rounds += 2

    def execute_decision(self, decision):
        self.candidate_list.append(index2(decision))


from numba.typed import Dict
from numba import types


@njit
def hash_board(chessboard):
    res0, res1 = types.uint64(0), types.uint64(0)
    c = chessboard.reshape(2, max_piece_cnt // 2)
    for i in c[0]:
        res0 = 3 * res0 + types.uint64(i + 1)
    for i in c[1]:
        res1 = 3 * res1 + types.uint64(i + 1)
    return res0, res1  # 为了避免超过int64。
    # res = ""
    # for i in chessboard.reshape(1, max_piece_cnt):
    #     res+=(i+1)
    # return res
    # return np.array2string(chessboard)


avg_breath = np.array([0., 0., 0., 0., 4., 3.,
                       4.625, 5.05, 5.75, 6.075, 7.125, 7.45,
                       8.05, 8.125, 8.975, 8.925, 9.1, 9.975,
                       9.725, 10.35, 11., 10.575, 11.525, 11.275,
                       11.4, 12.15, 11.525, 11.85, 12.275, 11.975,
                       12.425, 11.725, 12.9, 12.25, 12.975, 12.375,
                       12.75, 11.725, 12.175, 11.575, 12.075, 11.125,
                       11.6, 10.85, 10.75, 10.125, 10.5, 9.05,
                       9.5, 8.5, 8.475, 7.65, 7.5, 6.675,
                       6.95, 5.55, 5.3, 4.46341463, 4.15, 3.475,
                       2.825, 2.11904762, 1.43478261, 0.70175439, 0.])


@njit
def compute_breadth(rounds, depth):
    if 0 < rounds + depth <= len(avg_breath):
        return avg_breath[rounds + depth] / 2.5
    else:
        return 8


# @njit
def iterative_deepening_search(chessboard, current_color, time_out, rounds, decider, memory_out=1048576):
    """
    先搜低层的，保存值。如果剪枝了，下一次把没有值的放在最后，有值的放在前面。
    :param start_time:
    :param time_out:
    :param memory_out: 表示最多存多少个节点。1048576为1M节点。
    :return:
    """
    current_depth = 2
    assumed_breadth = compute_breadth(rounds, current_depth)
    # assumed_breadth = 10 if rounds <=5 else 2.5

    start_time, time_used = time.time(), 0
    # hash_table = Dict.empty(
    #     key_type=types.unicode_type,
    #     value_type=types.float64[:]  # 表示顺序？
    # )
    hash_table = Dict.empty(
        key_type=types.UniTuple(types.uint64, 2),
        value_type=types.float64  # 表示节点的价值。 如果哈希表有值，优先使用该值排序
    )  # 如果有LRU，可以把继承上一次的table
    value, move = -np.inf, None
    try:
        last_move = move
        while (time_used * assumed_breadth < time_out or (last_move != move).any()) and (
                time_used * 1.5 < time_out) and value != 1:  # 铁定合法的
            last_move = move
            value, move = alpha_beta_search(hash_table, start_time, time_out, memory_out, chessboard, current_color,
                                            current_depth, rounds)
            if move is not None:
                decider(move)
            current_depth += 1
            assumed_breadth = compute_breadth(rounds, current_depth)
            time_used = time.time() - start_time
        print(f"tried depth of {current_depth - 1}, there is still {time_out - time_used}s remained. ")
        # while value != 1:
        #     value, move = alpha_beta_search(hash_table, start_time, time_out, memory_out, chessboard, current_color,
        #                                     current_depth, rounds)
        #     current_depth += 1
        # print(f"Reached a search depth of {current_depth}")
        if (last_move == move).all():
            print("Do not go deeper because good move has achieved.")
        if value == 1:
            print("Value is 1! ")
    except TimeoutError:
        print("Timeout Error , but why?")
    finally:
        print(f"Reached a search depth of {current_depth - 1}")
        print(f"memory used: {len(hash_table)}/{memory_out}")
        return value, move


@njit
def insertion_sort(acts, new_chessboards, current_color, hash_table, rounds):
    """
    对 new_chessboards 按照简单评估函数，根据当前颜色的胜率进行从大到小排序。
    :param acts: 相应的action也要一起排序. 要求是list, 里面是np.array
    :param current_color: 当前的阵营。
    :param new_chessboards: 要排序的棋盘布局
    """
    new_chessboards_values = []
    for c in new_chessboards:
        v = hash_table.get(hash_board(c))
        if v is None:
            new_chessboards_values.append(组合策略1(c, current_color, rounds))
        else:
            # *current_color：存的是后是在存者视角下的优秀值*存者颜色，取的时候如果与存者颜色相等，就会抵消，否则会取反
            # 这种方法前提是评估函数具有正反对称性
            # +2 有哈希表的时候优先使用哈希表。如果没有，应该是被淘汰了。
            new_chessboards_values.append(v * current_color + 2)

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


@njit
def alpha_beta_search(hash_table, start_time, time_out, memory_out, chessboard, current_color, remaining_depth=6,
                      rounds=4,
                      alphas=np.array([-np.inf, -np.inf])):
    """

    :param rounds:
    :param hash_table: Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
    :param start_time:
    :param time_out:
    :param memory_out:

    :param chessboard:
    :param current_color:
    :param remaining_depth: 0 表示 直接对节点估值，不合法。 1表示一层贪心。 根据时间资源和回合数，请合理分配搜索深度。目前知道10层全回合OK的
    :param alphas: 0:到目前为止，路径上发现的 color=-1这个agent 的最佳选择值
                  1:到目前为止，路径上发现的 color= 1这个agent 的最佳选择值
    :return: 返回对于chessboard,color这个节点，它最大的选择值是多少，以及它选择了哪个子节点。
    """
    # if time.time() - start_time >= time_out:
    #     raise TimeoutError("Too deep, 速回！")
    alphas = alphas.copy()  # 防止修改上面的alphas
    if is_terminal(chessboard):
        utility = current_color * get_winner(chessboard)  # winner的颜色和我相等，就是1（颜色的平方性质）， 和我的颜色不等，就是-1.
        return symmetry_normalized_value(-1, 1, utility), None  # 满足截断性。由于其他价值函数也归一化了，-1和1就是最小值和最大值。 满足对手对称性。

    acts = typed.List(actions(chessboard, current_color))
    if len(acts) == 0:
        # 只能选择跳过这个action，value为对方的value
        value, move = alpha_beta_search(hash_table, start_time, time_out, memory_out, chessboard, -current_color,
                                        remaining_depth - 1, rounds + 1, alphas)
        value = -value
        if len(hash_table) < memory_out:
            hash_table[hash_board(chessboard)] = value * current_color
        return value, None  # 对手的值是和我反的。 我方没有action可以做。
    new_chessboards = typed.List(
        [updated_chessboard(chessboard, current_color, a) for a in acts])  # 用最多10倍内存换一半时间（排序和实际操作共用结果）
    insertion_sort(acts, new_chessboards, current_color, hash_table, rounds)

    if remaining_depth <= 1:  # 比如要求搜索1层，就是直接对max节点的所有邻接节点排序返回最大的。
        v = 组合策略1(new_chessboards[0], current_color, rounds)
        if v == 1:
            print("examine this!")
        return v, acts[0]  # 评价永远是根据我方的棋盘

    value, move = -np.inf, None  # 写在一起。每个节点都尝试让自己的价值最大化
    this_color_idx, other_color_idx = int((current_color + 1) // 2), int((-current_color + 1) // 2)
    for i, new_chessboard in enumerate(new_chessboards):
        action = acts[i]

        new_value, t = alpha_beta_search(hash_table, start_time, time_out, memory_out, new_chessboard, -current_color,
                                         remaining_depth - 1, rounds + 1, alphas)
        new_value = -new_value
        if len(hash_table) < memory_out:
            hash_table[hash_board(new_chessboard)] = new_value * current_color

        if new_value > value:
            value, move = new_value, action
            alphas[this_color_idx] = max(alphas[this_color_idx], value)
        # 另一种颜色的某一个节点已经到达了c = -beta的水平，低于c的都不接受。
        # 而我这个节点，至少可以达到v的水平。
        # 在那个对手节点看来，我至多会选择-v， 如果它自己的c已经比我这个-v大了，
        # 他就不会考虑我，我被剪枝，随便返回一个我的值和选择。
        if -value <= alphas[other_color_idx]:
            return value, move
    return value, move
