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
wins = 0
cnts = 0
while True:
    # for i_color in [ai.COLOR_BLACK, ai.COLOR_WHITE]:
    for i_color in [ai.COLOR_BLACK]:  # 对打随机应该一定胜利

        # 随机胜率
        agents = {i_color: new_ai.AI(chessboard_size, i_color, time_out),
                  -i_color: GreedyAI(chessboard_size, -i_color, random_baseline)}
        # 是否先手必胜  是
        # agents = {i_color: new_ai.AI(chessboard_size, i_color, time_out),
        #           -i_color: new_ai.AI(chessboard_size, -i_color, time_out)}
        simulator = Simulator(chessboard_size, time_out, agents)
        try:
            winner = simulator.quick_run(no_print=True)
        except Exception as e:
            print(e)
        else:
            if winner != i_color:
                print("loss")
                pass
            else:
                print("win")
                wins+=1
            cnts+=1
            print(wins/cnts)