import random
import unittest

import numpy as np

# from experiment.old_ai.AI import AI
# import src.project1.submit.AI as submit_ai
# from src.project1.submit.AI import AI
import src.project1.experimental.minimax迭代加深机 as experiment_ai

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0


class TestMoveValidity(unittest.TestCase):
    def __init__(self, methodName='TestMoveValidity'):
        super().__init__(methodName)
        self.chessboard_size = 8  # 是个int， 棋盘是 chessboard_size x chessboard_size
        # You are white or black
        self.color = COLOR_BLACK
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = 5
        self.ai = experiment_ai.AI(self.chessboard_size, self.color, self.time_out)

    def test_opening_is_valid(self):
        # self.assertEqual(True, False)  # add assertion here
        chessboard = np.zeros((8, 8))
        chessboard[(3, 3)] = COLOR_WHITE
        chessboard[(4, 4)] = COLOR_WHITE
        chessboard[(4, 3)] = COLOR_BLACK
        chessboard[(3, 4)] = COLOR_BLACK
        self.ai.go(chessboard)
        self.assertEqual({(2, 3), (3, 2), (5, 4), (4, 5)}, set(self.ai.candidate_list))

    def test_alpha_beta_bug(self):
        self.ai = experiment_ai.AI(self.chessboard_size, COLOR_WHITE, self.time_out)

        chessboard = np.array(
            [[-1, 1, 1, 1, 1, 1, 1, -1, ], [-1, 1, 1, 1, 1, 1, 1, -1, ], [-1, 1, -1, 1, -1, 1, 1, -1, ],
             [-1, 1, -1, 1, 1, -1, 1, -1, ], [-1, 1, 1, -1, -1, -1, 1, -1, ], [-1, -1, -1, -1, -1, -1, 1, -1, ],
             [0, 0, -1, -1, -1, 1, 1, -1, ], [0, -1, -1, -1, -1, -1, 1, -1, ]])
        self.ai.go(chessboard)

    def test_can_numba(self):
        chessboard = np.array(
            [[-1, 1, 1, 1, 1, 1, 1, -1, ], [-1, 1, 1, 1, 1, 1, 1, -1, ], [-1, 1, -1, 1, -1, 1, 1, -1, ],
             [-1, 1, -1, 1, 1, -1, 1, -1, ], [-1, 1, 1, -1, -1, -1, 1, -1, ], [-1, -1, -1, -1, -1, -1, 1, -1, ],
             [0, 0, -1, -1, -1, 1, 1, -1, ], [0, -1, -1, -1, -1, -1, 1, -1, ]])

        acts = experiment_ai.actions(chessboard, COLOR_WHITE)
        random.shuffle(acts)
        new_chessboards = [experiment_ai.updated_chessboard(chessboard, COLOR_WHITE, a) for a in acts]
        experiment_ai.insertion_sort(acts, new_chessboards, COLOR_WHITE)
        print(acts)
        # assert (acts == list(map(np.array, [(6, 0), (6, 1), (7, 0)]))).all()

    def test_ids_no_bug(self):
        self.ai = experiment_ai.AI(self.chessboard_size, COLOR_WHITE, np.inf)
        chessboard = np.array(
            [[-1, 1, 1, 1, 1, 1, 1, -1, ], [-1, 1, 1, 1, 1, 1, 1, -1, ], [-1, 1, -1, 1, -1, 1, 1, -1, ],
             [-1, 1, -1, 1, 1, -1, 1, -1, ], [-1, 1, 1, -1, -1, -1, 1, -1, ], [-1, -1, -1, -1, -1, -1, 1, -1, ],
             [0, 0, -1, -1, -1, 1, 1, -1, ], [0, -1, -1, -1, -1, -1, 1, -1, ]])
        self.ai.go(chessboard)

if __name__ == '__main__':
    unittest.main()
