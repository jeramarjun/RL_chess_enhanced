"""
Reinforcement Learning-Enhanced Chess Player (RL-AlphaBeta Agent).
Integrates the learned RL value function with Iterative-Deepening Negamax,
Alpha-Beta pruning, Quiescence search, Transposition Table, and Blitz clock management.
Fully conforms to the tournament agent interface (nodes_expanded, depth, get_best_move).
"""

import os
import time
import random
import numpy as np

from config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    EMPTY_SQUARE,
    BASE_PIECE_VALUES,
    DEFAULT_BLITZ_TIME,
)
from board import GameEngine, Move
from rl_value_net import LinearValueFunction


class RLEnhancedPlayer:
    """
    RL-Enhanced Chess Agent:
    Uses trained evaluation weights learned via TD-Leaf(lambda) self-play
    and combines them with a high-performance Alpha-Beta search engine.
    """

    INF = 1_000_000
    MATE = 900_000

    def __init__(self, board, weights_path=None, max_depth=6, time_limit=0.45):
        self.engine = board
        self.nodes_expanded = 0
        self.depth = 1

        # Search limits and clocks
        self.max_depth = max_depth
        self.default_time_limit = time_limit
        self.time_limit = time_limit
        self.start_time = 0.0
        self.time_up = False

        # Blitz clock management (60-second game clock)
        self.total_time_budget = DEFAULT_BLITZ_TIME
        self.safety_reserve = 2.0
        self.time_used = 0.0
        self.move_number = 0

        # Transposition table and move ordering heuristics
        self.transposition_table = {}  # key -> (depth, value, flag, best_move)
        self.killer_moves = {}         # ply -> [move_str, move_str]
        self.history = {}              # move_str -> score

        # RL Value Function
        self.value_fn = LinearValueFunction()
        if weights_path and os.path.exists(weights_path):
            self.value_fn.load_weights(weights_path)
            self.weights_source = weights_path
        else:
            default_weights = os.path.join(os.path.dirname(__file__), "trained_weights.json")
            if os.path.exists(default_weights):
                self.value_fn.load_weights(default_weights)
                self.weights_source = default_weights
            else:
                self.weights_source = "heuristic_prior"

        self.quiescence_depth = 3

    # -------------------------------------------------------------
    # Helper & Clock Management
    # -------------------------------------------------------------

    def _move_to_str(self, move):
        if not move:
            return ""
        return f"{move.start_row}{move.start_col}{move.end_row}{move.end_col}{move.promotion or ''}"

    def _position_key(self):
        cb = self.engine.chess_board
        return (cb.board_fen(), cb.turn, cb.castling_rights, cb.ep_square)

    def _time_exceeded(self):
        if (time.perf_counter() - self.start_time) >= self.time_limit:
            self.time_up = True
            return True
        return False

    def _allocate_time_budget(self, legal_moves):
        """Allocate time slice based on remaining Blitz clock and board phase."""
        remaining = self.total_time_budget - self.time_used - self.safety_reserve
        if remaining <= 0.20:
            return 0.01  # Emergency rapid move

        # Adaptive allocation: average 40 moves per game
        moves_left = max(10, 45 - self.move_number)
        base_slice = remaining / moves_left

        # Tactical urgency boost if in check or high capture density
        is_check = self.engine.is_in_check()
        num_captures = sum(1 for m in legal_moves if m.piece_captured != EMPTY_SQUARE)
        multiplier = 1.0
        if is_check:
            multiplier += 0.4
        if num_captures >= 3:
            multiplier += 0.2

        allocated = base_slice * multiplier
        # Cap per-move time to keep clock healthy
        ceiling = min(remaining * 0.30, 2.5)
        return max(0.02, min(allocated, ceiling))

    # -------------------------------------------------------------
    # Main Entry: get_best_move
    # -------------------------------------------------------------

    def get_best_move(self):
        """
        Calculates and returns the best move within time budget using
        iterative-deepening Negamax with Alpha-Beta pruning and learned RL evaluation.
        """
        self.start_time = time.perf_counter()
        self.time_up = False
        self.nodes_expanded = 0
        self.depth = 1

        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return None

        if len(legal_moves) == 1:
            time_spent = time.perf_counter() - self.start_time
            self.time_used += time_spent
            self.move_number += 1
            return legal_moves[0]

        # Allocate time
        self.time_limit = self._allocate_time_budget(legal_moves)

        # Quick fallback move: best 1-ply RL evaluation
        fallback_move = self._fast_ordered_move(legal_moves)
        best_move = fallback_move

        # Clear per-search heuristics (retain TT across positions)
        self.killer_moves.clear()

        # Iterative Deepening
        for current_depth in range(1, self.max_depth + 1):
            if self._time_exceeded():
                break

            self.depth = current_depth
            alpha = -self.INF
            beta = self.INF
            depth_best_move = None
            depth_best_score = -self.INF
            completed_depth = True

            # Root move ordering: prioritize previous best move
            ordered_moves = self._order_moves(legal_moves, ply=0, pv_move=best_move)

            for move in ordered_moves:
                if self._time_exceeded():
                    completed_depth = False
                    break

                self.engine.make_move(move)
                try:
                    score = -self._negamax(
                        depth=current_depth - 1,
                        alpha=-beta,
                        beta=-alpha,
                        ply=1
                    )
                finally:
                    self.engine.undo_move()

                if self.time_up:
                    completed_depth = False
                    break

                if score > depth_best_score:
                    depth_best_score = score
                    depth_best_move = move

                if score > alpha:
                    alpha = score

            if completed_depth and depth_best_move is not None:
                best_move = depth_best_move
                # If mate found, execute immediately
                if depth_best_score >= self.MATE - 100:
                    break

        time_spent = time.perf_counter() - self.start_time
        self.time_used += time_spent
        self.move_number += 1
        return best_move

    # -------------------------------------------------------------
    # Negamax with Alpha-Beta Pruning
    # -------------------------------------------------------------

    def _negamax(self, depth, alpha, beta, ply):
        self.nodes_expanded += 1

        if self._time_exceeded():
            return 0

        # Transposition Table Lookup
        key = self._position_key()
        alpha_orig = alpha
        cached = self.transposition_table.get(key)
        pv_move = None
        if cached is not None:
            c_depth, c_val, c_flag, c_move = cached
            pv_move = c_move
            if c_depth >= depth:
                if c_flag == "EXACT":
                    return c_val
                elif c_flag == "LOWER":
                    alpha = max(alpha, c_val)
                elif c_flag == "UPPER":
                    beta = min(beta, c_val)
                if alpha >= beta:
                    return c_val

        # Check terminal states
        state = self.engine.get_game_state()
        if state == "checkmate":
            return -self.MATE + ply
        if state in ("stalemate", "draw"):
            return 0

        # Leaf node: Quiescence Search
        if depth <= 0:
            return self._quiescence(alpha, beta, ply, qdepth=0)

        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            if self.engine.is_in_check():
                return -self.MATE + ply
            return 0

        ordered_moves = self._order_moves(legal_moves, ply=ply, pv_move=pv_move)
        best_val = -self.INF
        best_move = None

        for move in ordered_moves:
            if self._time_exceeded():
                return 0

            self.engine.make_move(move)
            try:
                val = -self._negamax(depth - 1, -beta, -alpha, ply + 1)
            finally:
                self.engine.undo_move()

            if self.time_up:
                return 0

            if val > best_val:
                best_val = val
                best_move = move

            if val > alpha:
                alpha = val

            if alpha >= beta:
                # Alpha-Beta Cutoff: Record killer and history heuristics
                self._record_cutoff(move, depth, ply)
                break

        # Store in Transposition Table
        if not self.time_up:
            if best_val <= alpha_orig:
                flag = "UPPER"
            elif best_val >= beta:
                flag = "LOWER"
            else:
                flag = "EXACT"
            self.transposition_table[key] = (depth, best_val, flag, best_move)

        return best_val

    # -------------------------------------------------------------
    # Quiescence Search
    # -------------------------------------------------------------

    def _quiescence(self, alpha, beta, ply, qdepth=0):
        self.nodes_expanded += 1

        if self._time_exceeded():
            return self._evaluate_position()

        # Stand-pat evaluation
        stand_pat = self._evaluate_position()

        if qdepth >= self.quiescence_depth:
            return stand_pat

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        legal_moves = self.engine.get_legal_moves()
        # Only search captures in quiescence
        captures = [m for m in legal_moves if m.piece_captured != EMPTY_SQUARE]
        if not captures:
            return stand_pat

        ordered_captures = sorted(captures, key=self._mvv_lva_score, reverse=True)

        for move in ordered_captures:
            if self._time_exceeded():
                return alpha

            # Delta pruning: skip if captured piece cannot possibly raise alpha
            cap_val = abs(BASE_PIECE_VALUES.get(move.piece_captured, 0))
            if stand_pat + cap_val + 150 < alpha:
                continue

            self.engine.make_move(move)
            try:
                val = -self._quiescence(-beta, -alpha, ply + 1, qdepth + 1)
            finally:
                self.engine.undo_move()

            if val >= beta:
                return beta
            if val > alpha:
                alpha = val

        return alpha

    # -------------------------------------------------------------
    # Evaluation: Learned RL Value Function
    # -------------------------------------------------------------

    def _evaluate_position(self):
        """
        Evaluate current state from side-to-move perspective using the
        learned RL Value Function:
          score = 1000.0 * V_theta_side(s)
        """
        side_val = self.value_fn.evaluate_side(self.engine)
        return int(round(1000.0 * side_val))

    def evaluate_board(self, game_state=None):
        """
        External evaluation compatibility method.
        Returns White-perspective evaluation in centipawns.
        """
        v_white = self.value_fn.evaluate_state(self.engine)
        return round(1000.0 * v_white, 2)

    # -------------------------------------------------------------
    # Move Ordering & Heuristics
    # -------------------------------------------------------------

    def _mvv_lva_score(self, move):
        victim = abs(BASE_PIECE_VALUES.get(move.piece_captured, 0))
        attacker = abs(BASE_PIECE_VALUES.get(move.piece_moved, 0))
        return victim * 10 - attacker

    def _order_moves(self, moves, ply, pv_move=None):
        def score(m):
            if pv_move is not None and m == pv_move:
                return 1_000_000.0

            s = 0.0
            # Captures via MVV-LVA
            if m.piece_captured != EMPTY_SQUARE:
                s += 100_000.0 + self._mvv_lva_score(m)

            # Promotions
            if m.promotion:
                s += 80_000.0

            # Killer moves
            m_str = self._move_to_str(m)
            killers = self.killer_moves.get(ply, [])
            if m_str in killers:
                s += 20_000.0 - killers.index(m_str) * 2000.0

            # History heuristic
            s += self.history.get(m_str, 0.0)

            # Center development bonus
            if m.end_row in (3, 4) and m.end_col in (3, 4):
                s += 50.0

            return s

        return sorted(moves, key=score, reverse=True)

    def _record_cutoff(self, move, depth, ply):
        m_str = self._move_to_str(move)
        # History
        self.history[m_str] = self.history.get(m_str, 0.0) + (depth * depth)

        # Killers
        killers = self.killer_moves.setdefault(ply, [])
        if m_str not in killers:
            killers.insert(0, m_str)
            if len(killers) > 2:
                killers.pop()

    def _fast_ordered_move(self, legal_moves):
        """Quick 1-ply RL evaluation move as immediate fallback."""
        best_m = legal_moves[0]
        best_s = -self.INF
        for m in legal_moves:
            self.engine.make_move(m)
            # Side to move after make_move is opponent, so negate
            s = -self._evaluate_position()
            self.engine.undo_move()
            if s > best_s:
                best_s = s
                best_m = m
        return best_m
