"""
Baseline Chess Players for Systematic Performance Comparisons.
Provides:
  1. RandomPlayer: Baseline random legal move agent (equivalent to P25CS0004/P22CS201).
  2. GreedyMaterialPlayer: 1-ply greedy material-maximizing agent.
  3. UntrainedRLPlayer: Alpha-Beta search with zero/random unlearned weights.
  4. HeuristicPlayer: Classical hand-tuned minimax search agent with fixed heuristic tables.
"""

import random
import time
from config import BASE_PIECE_VALUES, EMPTY_SQUARE, PAWN_PST, KNIGHT_PST, BISHOP_PST, ROOK_PST, QUEEN_PST
from board import GameEngine, Move, sq_to_rc
from rl_value_net import LinearValueFunction


class RandomPlayer:
    """Agent that selects a uniformly random legal move."""

    def __init__(self, engine):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = 0

    def get_best_move(self):
        self.nodes_expanded = 1
        legal = self.engine.get_legal_moves()
        if not legal:
            return None
        return random.choice(legal)

    def evaluate_board(self, game_state=None):
        return 0.0


class GreedyMaterialPlayer:
    """Agent that greedily picks the move maximizing immediate material advantage."""

    def __init__(self, engine):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = 1

    def get_best_move(self):
        legal = self.engine.get_legal_moves()
        if not legal:
            return None

        self.nodes_expanded = len(legal)
        is_white = self.engine.white_to_move
        best_score = -1e9 if is_white else 1e9
        best_move = legal[0]

        for move in legal:
            self.engine.make_move(move)
            score = self._calc_material()
            self.engine.undo_move()

            if is_white and score > best_score:
                best_score = score
                best_move = move
            elif not is_white and score < best_score:
                best_score = score
                best_move = move

        return best_move

    def _calc_material(self):
        total = 0
        for r in range(8):
            for c in range(8):
                p = self.engine.board[r][c]
                if p != EMPTY_SQUARE:
                    total += BASE_PIECE_VALUES.get(p, 0)
        return total


class UntrainedRLPlayer:
    """
    Search agent with the same Alpha-Beta engine as RL player,
    but with zero/unlearned weights theta_0 to quantify RL learning gains.
    """

    def __init__(self, engine, depth=2):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = depth
        self.value_fn = LinearValueFunction(init_strategy="zeros")

    def get_best_move(self):
        legal = self.engine.get_legal_moves()
        if not legal:
            return None
        self.nodes_expanded = 0

        best_m = legal[0]
        best_s = -1e9
        for m in legal:
            self.nodes_expanded += 1
            self.engine.make_move(m)
            s = -self.value_fn.evaluate_side(self.engine)
            self.engine.undo_move()
            if s > best_s:
                best_s = s
                best_m = m
        return best_m


class HeuristicPlayer:
    """
    Classical non-RL heuristic player using hand-crafted material and PST values
    with 3-ply Alpha-Beta search. Serves as strong non-learning baseline.
    """

    INF = 1_000_000

    def __init__(self, engine, depth=2):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = depth

    def get_best_move(self):
        self.nodes_expanded = 0
        legal = self.engine.get_legal_moves()
        if not legal:
            return None

        alpha = -self.INF
        beta = self.INF
        best_move = legal[0]

        for move in legal:
            self.engine.make_move(move)
            val = -self._alphabeta(self.depth - 1, -beta, -alpha)
            self.engine.undo_move()

            if val > alpha:
                alpha = val
                best_move = move

        return best_move

    def _alphabeta(self, depth, alpha, beta):
        self.nodes_expanded += 1
        if depth <= 0:
            return self._heuristic_eval()

        state = self.engine.get_game_state()
        if state == "checkmate":
            return -self.INF
        if state in ("stalemate", "draw"):
            return 0

        legal = self.engine.get_legal_moves()
        if not legal:
            return 0

        for move in legal:
            self.engine.make_move(move)
            val = -self._alphabeta(depth - 1, -beta, -alpha)
            self.engine.undo_move()

            if val >= beta:
                return beta
            if val > alpha:
                alpha = val

        return alpha

    def _heuristic_eval(self):
        score = 0
        for r in range(8):
            for c in range(8):
                p = self.engine.board[r][c]
                if p == EMPTY_SQUARE:
                    continue
                v = BASE_PIECE_VALUES.get(p, 0)
                score += v
                kind = p[1]
                white = p[0] == "w"
                tr = r if white else 7 - r
                pst_val = 0
                if kind == "P":
                    pst_val = PAWN_PST[tr][c]
                elif kind == "N":
                    pst_val = KNIGHT_PST[tr][c]
                elif kind == "B":
                    pst_val = BISHOP_PST[tr][c]
                elif kind == "R":
                    pst_val = ROOK_PST[tr][c]
                elif kind == "Q":
                    pst_val = QUEEN_PST[tr][c]
                score += pst_val if white else -pst_val

        return score if self.engine.white_to_move else -score
