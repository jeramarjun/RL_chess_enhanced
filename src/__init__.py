"""
Reinforcement Learning Chess Engine Source Package.
Exposes core board rules, feature extraction, RL value functions, TD learning,
Alpha-Beta player, baselines, and blitz game runner.
"""

from .config import *
from .board import GameEngine, Move, sq_to_rc, rc_to_sq
from .features import FeatureExtractor, FEATURE_DIM, FEATURE_NAMES
from .rl_value_net import LinearValueFunction, NeuralValueNetwork
from .td_learning import TDTrainer
from .rl_player import RLEnhancedPlayer
from .baselines import RandomPlayer, GreedyMaterialPlayer, UntrainedRLPlayer, HeuristicPlayer
from .game_runner import run_blitz_game, display_board, PlayerClock

__all__ = [
    "GameEngine",
    "Move",
    "sq_to_rc",
    "rc_to_sq",
    "FeatureExtractor",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "LinearValueFunction",
    "NeuralValueNetwork",
    "TDTrainer",
    "RLEnhancedPlayer",
    "RandomPlayer",
    "GreedyMaterialPlayer",
    "UntrainedRLPlayer",
    "HeuristicPlayer",
    "run_blitz_game",
    "display_board",
    "PlayerClock",
]
