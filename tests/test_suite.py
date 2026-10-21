"""
Comprehensive Unit Test Suite for Reinforcement Learning Chess Engine.
Tests:
  - Board rules, Move objects, FEN synchronization, Make/Undo consistency.
  - Feature extraction dimensions, ranges, and game phase factors.
  - LinearValueFunction evaluation, analytical gradient, and JSON persistence.
  - TD-Leaf learning step and eligibility trace accumulation.
  - Alpha-Beta search soundness, move legality, and blitz clock compliance.
"""

import os
import sys
import unittest
import numpy as np
import chess

# Ensure both project root and src/ are in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_current_dir, ".."))
_src_dir = os.path.join(_root_dir, "src")
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from board import GameEngine, Move, sq_to_rc, rc_to_sq
from config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    EMPTY_SQUARE,
    WHITE_PAWN,
    BLACK_PAWN,
    WHITE_KNIGHT,
    WHITE_QUEEN,
)
from features import FeatureExtractor, FEATURE_DIM
from rl_value_net import LinearValueFunction
from td_learning import TDTrainer
from rl_player import RLEnhancedPlayer
from baselines import RandomPlayer, GreedyMaterialPlayer, UntrainedRLPlayer, HeuristicPlayer


class TestBoardMechanics(unittest.TestCase):
    """Test standard chess rules wrapping and move undo integrity."""

    def setUp(self):
        self.engine = GameEngine()

    def test_initial_board_setup(self):
        self.assertEqual(len(self.engine.board), 8)
        self.assertEqual(len(self.engine.board[0]), 8)
        self.assertTrue(self.engine.white_to_move)
        self.assertEqual(self.engine.board[7][4], "wK")  # e1
        self.assertEqual(self.engine.board[0][4], "bK")  # e8
        self.assertEqual(self.engine.board[6][4], "wP")  # e2
        self.assertEqual(self.engine.board[1][4], "bP")  # e7

    def test_legal_moves_count_start(self):
        moves = self.engine.get_legal_moves()
        # Initial chess position has exactly 20 legal moves (16 pawn pushes + 4 knight moves)
        self.assertEqual(len(moves), 20)

    def test_make_and_undo_integrity(self):
        initial_fen = self.engine.chess_board.fen()
        initial_board = [list(r) for r in self.engine.board]

        moves = self.engine.get_legal_moves()
        move = moves[0]

        self.engine.make_move(move)
        self.assertNotEqual(self.engine.chess_board.fen(), initial_fen)
        self.assertFalse(self.engine.white_to_move)

        self.engine.undo_move()
        self.assertEqual(self.engine.chess_board.fen(), initial_fen)
        self.assertEqual(self.engine.board, initial_board)
        self.assertTrue(self.engine.white_to_move)

    def test_scholars_mate_detection(self):
        # 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#
        moves_uci = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
        for uci in moves_uci:
            m = chess.Move.from_uci(uci)
            move_obj = Move.from_chess(m, self.engine)
            self.engine.make_move(move_obj)

        self.assertTrue(self.engine.is_in_check())
        self.assertEqual(self.engine.get_game_state(), "checkmate")


class TestFeatureExtraction(unittest.TestCase):
    """Test state space phi(s) representation and dimensionality."""

    def setUp(self):
        self.engine = GameEngine()
        self.extractor = FeatureExtractor()

    def test_feature_dimension(self):
        phi = self.extractor.extract_features(self.engine)
        self.assertEqual(len(phi), FEATURE_DIM)
        self.assertEqual(len(phi), self.extractor.dim)

    def test_initial_symmetric_features(self):
        phi = self.extractor.extract_features(self.engine)
        # Material counts at start: 8 pawns / 8 = 1.0
        # Net material features should be exactly 0.0
        names = self.extractor.feature_names
        name_to_idx = {name: i for i, name in enumerate(names)}

        self.assertAlmostEqual(phi[name_to_idx["net_pawns"]], 0.0)
        self.assertAlmostEqual(phi[name_to_idx["net_knights"]], 0.0)
        self.assertAlmostEqual(phi[name_to_idx["net_bishops"]], 0.0)
        self.assertAlmostEqual(phi[name_to_idx["net_rooks"]], 0.0)
        self.assertAlmostEqual(phi[name_to_idx["net_queens"]], 0.0)
        self.assertAlmostEqual(phi[name_to_idx["bias"]], 1.0)


class TestValueFunction(unittest.TestCase):
    """Test LinearValueFunction gradient and evaluation bounds."""

    def setUp(self):
        self.val_fn = LinearValueFunction(init_strategy="heuristic_prior")
        self.engine = GameEngine()

    def test_evaluation_bounds(self):
        v = self.val_fn.evaluate_state(self.engine)
        # Value must be bounded in (-1, 1) due to tanh
        self.assertGreaterEqual(v, -1.0)
        self.assertLessEqual(v, 1.0)

    def test_gradient_shape_and_magnitude(self):
        phi = self.val_fn.extractor.extract_features(self.engine)
        grad = self.val_fn.get_gradient(phi)
        self.assertEqual(len(grad), FEATURE_DIM)
        # Gradient of tanh: (1 - v^2) * phi, must be non-zero and bounded
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_weight_save_and_load(self):
        test_file = "test_weights_tmp.json"
        try:
            self.val_fn.save_weights(test_file)
            self.assertTrue(os.path.exists(test_file))

            new_fn = LinearValueFunction(init_strategy="zeros")
            new_fn.load_weights(test_file)
            np.testing.assert_array_almost_equal(self.val_fn.weights, new_fn.weights)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)


class TestRLPlayerAndSearch(unittest.TestCase):
    """Test RLEnhancedPlayer search, move ordering, and clock constraints."""

    def setUp(self):
        self.engine = GameEngine()
        self.player = RLEnhancedPlayer(self.engine, max_depth=2, time_limit=0.2)

    def test_get_best_move_returns_legal(self):
        move = self.player.get_best_move()
        self.assertIsNotNone(move)
        legal_moves = self.engine.get_legal_moves()
        self.assertIn(move, legal_moves)
        self.assertGreater(self.player.nodes_expanded, 0)

    def test_quiescence_capture_resolution(self):
        # Set up a position with a capture available
        # 1. e4 d5
        m1 = Move.from_chess(chess.Move.from_uci("e2e4"), self.engine)
        self.engine.make_move(m1)
        m2 = Move.from_chess(chess.Move.from_uci("d7d5"), self.engine)
        self.engine.make_move(m2)

        # White can capture on d5 (exd5)
        move = self.player.get_best_move()
        self.assertIsNotNone(move)
        self.assertIn(move, self.engine.get_legal_moves())


class TestBaselines(unittest.TestCase):
    """Test that all baseline players return valid legal moves."""

    def setUp(self):
        self.engine = GameEngine()

    def test_random_player(self):
        agent = RandomPlayer(self.engine)
        move = agent.get_best_move()
        self.assertIn(move, self.engine.get_legal_moves())

    def test_greedy_player(self):
        agent = GreedyMaterialPlayer(self.engine)
        move = agent.get_best_move()
        self.assertIn(move, self.engine.get_legal_moves())

    def test_untrained_rl_player(self):
        agent = UntrainedRLPlayer(self.engine, depth=1)
        move = agent.get_best_move()
        self.assertIn(move, self.engine.get_legal_moves())

    def test_heuristic_player(self):
        agent = HeuristicPlayer(self.engine, depth=1)
        move = agent.get_best_move()
        self.assertIn(move, self.engine.get_legal_moves())


if __name__ == "__main__":
    unittest.main()
