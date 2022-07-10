import src.project1.submit.AI as ai


class GreedyAI(object):
    def __init__(self, chessboard_size, color, evaluator):
        self.chessboard_size = chessboard_size  # 是个int， 棋盘是 chessboard_size x chessboard_size
        self.color = color  # 黑或白
        self.reversi_env = ai.ReversiEnv(chessboard_size)
        self.evaluator = evaluator
        self.candidate_list = []
        self.chessboard = None

    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        self.chessboard = chessboard

        self.candidate_list = self.reversi_env.actions(chessboard, self.color, self.candidate_list)
        if len(self.candidate_list) == 0:
            return
        else:
            decision = max(self.candidate_list, key=lambda p: self.evaluator(self, p))
            self.execute_decision(decision)

    def execute_decision(self, decision):
        self.candidate_list.append(decision)
