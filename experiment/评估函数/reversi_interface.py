class Evaluator:
    def __init__(self, env=None, chessboard_size=None):
        self.env = env
        self.chessboard_size = chessboard_size
        self.chessboard = None
        self.color = None
    def set_env(self, env):
        self.env = env
    def set_color(self, color):
        self.color = color
    def set_chessboard_size(self, chessboard_size):
        self.chessboard_size = chessboard_size
    def set_chessboard(self, chessboard):
        raise NotImplementedError
    def evaluate(self, piece):
        raise NotImplementedError