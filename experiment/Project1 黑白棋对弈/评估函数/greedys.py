from experiment.评估函数.simulator import Simulator
from experiment.评估函数.greedy_ai import GreedyAI
import experiment.old_ai.AI as ai
from numba import njit

import numpy as np
import random


def random_baseline(a, p):
    return random.random()  # 0-1随机权重


# action greedy
def middle_action_first(a, p):
    return -np.linalg.norm(np.array(p) - np.array([a.chessboard_size / 2, a.chessboard_size / 2]))


Vmap = np.array([[500, -25, 10, 5, 5, 10, -25, 500],
                 [-25, -45, 1, 1, 1, 1, -45, -25],
                 [10, 1, 3, 2, 2, 3, 1, 10],
                 [5, 1, 2, 1, 1, 2, 1, 5],
                 [5, 1, 2, 1, 1, 2, 1, 5],
                 [10, 1, 3, 2, 2, 3, 1, 10],
                 [-25, -45, 1, 1, 1, 1, -45, -25],
                 [500, -25, 10, 5, 5, 10, -25, 500]])


def map_weight_sum(a, p):
    new_chessboard = a.reversi_env.updated_chessboard(a.chessboard, a.color, p)
    return _map_weight_sum(new_chessboard, a.color)


@njit(cache=True)
def _map_weight_sum(board, mycolor):
    return -(board * Vmap).sum() * mycolor


Vmap2 = np.ones((8, 8))


def map_weight_sum2(a, p):
    new_chessboard = a.reversi_env.updated_chessboard(a.chessboard, a.color, p)
    return _map_weight_sum2(new_chessboard, a.color)


@njit(cache=True)
def _map_weight_sum2(board, mycolor):
    return -(board * Vmap2).sum() * mycolor


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