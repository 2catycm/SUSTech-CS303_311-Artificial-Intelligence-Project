import unittest

import numpy as np

from AI import AI

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0


class MyTestCase(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.chessboard_size = 8  # 是个int， 棋盘是 chessboard_size x chessboard_size
        # You are white or black
        self.color = COLOR_BLACK
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = 5
        self.ai = AI(self.chessboard_size, self.color, self.time_out)


    def test_start_config(self):
        # self.assertEqual(True, False)  # add assertion here
        chessboard = np.zeros((8,8))
        chessboard[(3, 3)] = COLOR_WHITE
        chessboard[(4, 4)] = COLOR_WHITE
        chessboard[(4, 3)] = COLOR_BLACK
        chessboard[(3, 4)] = COLOR_BLACK
        self.ai.find_valid(chessboard)
        self.assertEqual({(2, 3), (3, 2), (5, 4), (4, 5)}, set(self.ai.candidate_list))


if __name__ == '__main__':
    unittest.main()
