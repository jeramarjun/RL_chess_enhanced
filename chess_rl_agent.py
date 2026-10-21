"""
Course Submission AI Agent Wrapper:
Exposes the trained RL-Enhanced Chess Engine under standard submission conventions.
Instantiates RLEnhancedPlayer with trained weights and Alpha-Beta search.
"""

import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_current_dir, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

try:
    from rl_player import RLEnhancedPlayer
except ImportError:
    from src.rl_player import RLEnhancedPlayer


class ChessRLAgent(RLEnhancedPlayer):
    """
    Final submitted Reinforcement Learning Chess Agent.
    Subclasses RLEnhancedPlayer with automatic trained weight loading.
    """

    def __init__(self, board):
        # Look for weights in root or models/
        weights_candidates = [
            os.path.join(_current_dir, "trained_weights.json"),
            os.path.join(_current_dir, "models", "trained_weights.json"),
        ]
        chosen_weights = "trained_weights.json"
        for w in weights_candidates:
            if os.path.exists(w):
                chosen_weights = w
                break

        super().__init__(board, weights_path=chosen_weights, max_depth=6, time_limit=0.45)


# Alias class names for tournament runners
RLPlayer = ChessRLAgent
B24CM_RL = ChessRLAgent
AIPlayer = ChessRLAgent
B24CM1034 = ChessRLAgent
