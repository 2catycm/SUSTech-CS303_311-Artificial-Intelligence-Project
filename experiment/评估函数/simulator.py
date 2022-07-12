import numpy as np

# import src.project1.submit.AI as ai
import experiment.old_ai.AI as ai

import matplotlib.pyplot as plt


class Simulator:
    @staticmethod
    def init_chessboard(chessboard_size):
        chessboard = np.zeros((chessboard_size, chessboard_size))
        color = ai.COLOR_BLACK
        locations = [chessboard_size // 2 - 1, chessboard_size // 2]
        for i in range(2):
            for j in range(2):
                j ^= i  # 如果i是0， j不变，如果i是1，j反置
                chessboard[locations[i], locations[j]] = color
                color *= -1
        return chessboard

    def __init__(self, chessboard_size, time_out, agents, current_color=ai.COLOR_BLACK):
        self.chessboard_size = chessboard_size
        self.time_out = time_out
        self.reversi_env = ai.ReversiEnv(chessboard_size)
        self.current_color = current_color
        self.current_chessboard = Simulator.init_chessboard(chessboard_size)
        self.agents = agents  # color 映射 对象 的dict
        self.rounds = 4

    def quick_run(self, accountants=None, no_print=False):
        if accountants is None:
            accountants = []
        self.rounds = 4
        while not self.reversi_env.is_terminal(self.current_chessboard, self.current_color):
            if not no_print:
                print(f"Round {self.rounds}")
            # 注意落子空闲，我们定义没有行动力的时候回合不会增加
            black, white = self.reversi_env.piece_cnt(self.current_chessboard)
            if self.rounds!= black+white:
                raise AssertionError

            for accountant in accountants:
                accountant(self.rounds, self.current_color, self.current_chessboard, self.agents)
            self.quick_step()
            # yield self.current_chessboard
            # plt.imshow(self.current_chessboard)
            # plt.pause(0.01)
            # plt.show()

        return self.reversi_env.get_winner(self.current_chessboard)

    def quick_step(self):
        agent = self.agents[self.current_color]
        agent.go(self.current_chessboard)
        if len(agent.candidate_list) != 0:
            self.current_chessboard = self.reversi_env.updated_chessboard(self.current_chessboard, self.current_color,
                                                                          agent.candidate_list[-1])
            self.rounds += 1
        else:
            print("过")

        self.current_color *= -1
