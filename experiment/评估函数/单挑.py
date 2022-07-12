from simulator import Simulator
from greedy_ai import GreedyAI
import experiment.old_ai.AI as ai
from numba import njit

import src.project1.submit.AI as new_ai

import numpy as np
import random

chessboard_size = 4
time_out = 5


def random_baseline(a, p):
    return random.random()  # 0-1随机权重


# for i in range(100):
while True:
    for i_color in [ai.COLOR_BLACK, ai.COLOR_WHITE]:

        agents = {i_color: new_ai.AI(chessboard_size, i_color, time_out),
                  -i_color: GreedyAI(chessboard_size, -i_color, random_baseline)}
        simulator = Simulator(chessboard_size, time_out, agents)
        try:
            winner = simulator.quick_run()
        except Exception as e:
            print(e)
        else:
            print(winner)
