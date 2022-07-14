from simulator import Simulator
from greedy_ai import GreedyAI
import experiment.old_ai.AI as ai
from numba import njit

import src.project1.submit.AI as new_ai
import src.project1.experimental.minimax迭代加深机 as expermental_ai

import numpy as np
from greedys import *

# chessboard_size = 4
chessboard_size = 8
time_out = 5

# for i in range(100):
wins = 0
cnts = 0
while True:
    # for i_color in [ai.COLOR_BLACK, ai.COLOR_WHITE]:
    #     for i_color in [ai.COLOR_BLACK]:  # 对打随机应该一定胜利
    for i_color in [ai.COLOR_WHITE]:  # 对打随机应该一定胜利

        # 4赛罗12层AI的随机胜率实验
        # agents = {i_color: new_ai.AI(chessboard_size, i_color, time_out),
        #           -i_color: GreedyAI(chessboard_size, -i_color, random_baseline)}
        # 是否先手必胜  是
        # agents = {i_color: new_ai.AI(chessboard_size, i_color, time_out),
        #           -i_color: new_ai.AI(chessboard_size, -i_color, time_out)}

        # 贪心算法随机胜率测试（正常黑白棋）
        # agents = {i_color: GreedyAI(chessboard_size, i_color, become_less_first),
        #           -i_color: GreedyAI(chessboard_size, -i_color, random_baseline)}
        # 实验结果：random_baseline 0.5 middle_action_first 0.7483 eat_less_first 0.615835
        # become_less_first 0.6118 map_weight_sum胜率 0.8277  map_weight_sum2 0.6122
        # 与对局函数的评分具有一致性（排序一致）：[3, 8, 3, 2, 10, 3]

        # ab剪枝的胜率，我们要离线训练还是在线训练？
        # ab剪枝 5层， 胜率约为0.769

        agents = {i_color: expermental_ai.AI(chessboard_size, i_color, time_out),
                  -i_color: GreedyAI(chessboard_size, -i_color, random_baseline)}  # 0.8

        simulator = Simulator(chessboard_size, time_out, agents)
        try:
            winner = simulator.quick_run(no_print=False)
            # winner = simulator.quick_run(no_print=True)
        except Exception as e:
            print(e)
        else:
            if winner != i_color:
                print("loss")
                pass
            else:
                print("win")
                wins += 1
            cnts += 1
            print(wins / cnts)
