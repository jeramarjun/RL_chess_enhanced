"""
Course Submission AI Agent Wrapper:
Exposes the trained RL-Enhanced Chess Engine under standard submission conventions.
Instantiates RLEnhancedPlayer with trained weights and Alpha-Beta search.
"""

from rl_player import RLEnhancedPlayer


class ChessRLAgent(RLEnhancedPlayer):
    """
    Final submitted Reinforcement Learning Chess Agent.
    Subclasses RLEnhancedPlayer with automatic trained weight loading.
    """

    def __init__(self, board):
        super().__init__(board, weights_path="trained_weights.json", max_depth=6, time_limit=0.45)


# Alias class names for tournament runners
RLPlayer = ChessRLAgent
B24CM_RL = ChessRLAgent
AIPlayer = ChessRLAgent
