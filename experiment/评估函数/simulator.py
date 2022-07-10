import numpy as np

import src.project1.submit.AI as ai

import matplotlib.pyplot as plt


class Simulator:
    @staticmethod
    def init_chessboard(chessboard_size):
        chessboard = np.zeros((chessboard_size, chessboard_size))
        color = ai.COLOR_BLACK
        for i in [chessboard_size / 2, chessboard_size / 2 + 1]:
            for j in [chessboard_size / 2, chessboard_size / 2 + 1]:
                chessboard[i, j] = color
                color *= -1
        return chessboard

    def __init__(self, chessboard_size, time_out, black_agent, white_agent, current_color=ai.COLOR_BLACK):
        self.chessboard_size = chessboard_size
        self.time_out = time_out
        self.reversi_env = ai.ReversiEnv(chessboard_size)
        self.current_color = current_color
        self.current_chessboard = Simulator.init_chessboard(chessboard_size)
        self.agents = {ai.COLOR_BLACK: black_agent, ai.COLOR_WHITE: white_agent}

    def quick_run(self):
        while not self.reversi_env.is_terminal(self.current_chessboard, self.current_color):
            self.quick_step()
            # yield self.current_chessboard
            # plt.imshow(self.current_chessboard)
        return self.reversi_env.get_winner(self.current_chessboard)

    def quick_step(self):
        agent = self.agents[self.current_color]
        agent.go(self.current_chessboard)
        self.current_chessboard = self.reversi_env.updated_chessboard(self.current_chessboard, self.current_color,
                                                                      agent.candidate_list[-1])
        self.current_color *= -1
