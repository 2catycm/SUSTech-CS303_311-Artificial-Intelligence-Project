import src.project1.submit.AI as ai
from reversi_interface import Evaluator


class GreedyEvaluator(Evaluator):

    def __init__(self, fun):
        super().__init__()
        self.fun = fun

    def set_chessboard(self, chessboard):
        self.chessboard = chessboard

    def evaluate(self, piece):
        return self.fun(self, piece)


class GreedyAI(object):
    def __init__(self, chessboard_size, color, evaluator):
        self.chessboard_size = chessboard_size  # 是个int， 棋盘是 chessboard_size x chessboard_size
        self.color = color  # 黑或白
        self.reversi_env = ai.ReversiEnv(chessboard_size)
        evaluator.set_color(self.color)
        evaluator.set_env(self.reversi_env)
        evaluator.set_chessboard_size(self.chessboard_size)
        self.evaluator = evaluator
        self.candidate_list = []

    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        self.candidate_list = self.reversi_env.actions(chessboard, self.color, self.candidate_list)
        if len(self.candidate_list) == 0:
            return
        else:
            self.evaluator.set_chessboard(chessboard)
            decision = max(self.candidate_list, key=self.evaluator.evaluate)
            self.execute_decision(decision)

    def execute_decision(self, decision):
        self.candidate_list.append(decision)
