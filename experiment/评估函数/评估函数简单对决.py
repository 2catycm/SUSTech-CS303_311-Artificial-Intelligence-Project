from simulator import Simulator
from greedy_ai import GreedyEvaluator
from greedy_ai import GreedyAI
import src.project1.submit.AI as ai

import numpy as np

chessboard_size = 8
time_out = 5


def middle_action_first(e, p):
    return -np.linalg.norm(np.array(p) - np.array([e.chessboard_size / 2, e.chessboard_size / 2]))
def eat_less_first(e, p):
    old_black, old_white = e.env.piece_cnt(e.chessboard)
    new_chessboard = e.env.updated_chessboard(chessboard, e.color)

greedy_functions = [middle_action_first, ]
greedy_evaluators = list(map(GreedyEvaluator, greedy_functions))
length = len(greedy_evaluators)
scores = [0 for i in range(length)]
for i in range(length):
    for j in range(i + 1, length):
        simulator = Simulator(chessboard_size, time_out,
                              GreedyAI(chessboard_size, ai.COLOR_BLACK, greedy_evaluators[i]),
                              GreedyAI(chessboard_size, ai.COLOR_WHITE, greedy_evaluators[j]))
        winner = simulator.quick_run()
        if winner == ai.COLOR_BLACK:
            scores[i]+=1
        elif winner == ai.COLOR_WHITE:
            scores[j]+=1
print(scores)
