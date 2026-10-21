"""
Temporal Difference Learning engine (TD-Leaf(lambda) / TD(lambda)) with Self-Play.
Trains the RL evaluation function weights through self-play experience,
temporal difference errors, and eligibility traces.
"""

import copy
import math
import random
import time
import numpy as np
import chess

from board import GameEngine, Move
from config import (
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA,
    DEFAULT_ALPHA,
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_MIN_EPSILON,
    DEFAULT_EPISODES,
    EMPTY_SQUARE,
    BASE_PIECE_VALUES,
)
from rl_value_net import LinearValueFunction


class TDTrainer:
    """
    Reinforcement Learning Trainer using TD-Leaf(lambda) and Self-Play.
    Optimizes evaluation weights theta based on game outcomes and rewards.
    """

    def __init__(
        self,
        value_fn=None,
        gamma=DEFAULT_GAMMA,
        lam=DEFAULT_LAMBDA,
        alpha=DEFAULT_ALPHA,
        epsilon=DEFAULT_EPSILON,
        epsilon_decay=DEFAULT_EPSILON_DECAY,
        min_epsilon=DEFAULT_MIN_EPSILON,
        search_depth=1,
    ):
        self.value_fn = value_fn if value_fn is not None else LinearValueFunction()
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.search_depth = search_depth

        # Training history logs
        self.history = {
            "episodes": [],
            "td_losses": [],
            "outcomes": [],        # +1 White win, -1 Black win, 0 Draw
            "game_lengths": [],    # plies
            "epsilons": [],
            "weight_history": [],  # snapshots of feature weights
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
        }

    def _select_action_and_leaf(self, engine, depth=1, epsilon=0.0):
        """
        Evaluate legal moves via minimax search to find best move and principal leaf.
        Returns (best_move, leaf_engine_copy).
        Exploration via epsilon-greedy selection.
        """
        legal_moves = engine.get_legal_moves()
        if not legal_moves:
            return None, engine.copy()

        # Epsilon-greedy exploration
        if random.random() < epsilon:
            chosen_move = random.choice(legal_moves)
            # Find leaf by playing chosen move
            engine.make_move(chosen_move)
            leaf = engine.copy()
            engine.undo_move()
            return chosen_move, leaf

        is_white = engine.white_to_move
        best_move = legal_moves[0]
        best_val = -1e9 if is_white else 1e9
        best_leaf = None

        # Shallow 1-ply search with leaf evaluation
        for move in legal_moves:
            engine.make_move(move)
            val = self.value_fn.evaluate_state(engine)

            if is_white:
                if val > best_val:
                    best_val = val
                    best_move = move
                    best_leaf = engine.copy()
            else:
                if val < best_val:
                    best_val = val
                    best_move = move
                    best_leaf = engine.copy()

            engine.undo_move()

        if best_leaf is None:
            best_leaf = engine.copy()

        return best_move, best_leaf

    def play_self_play_game(self, max_plies=120):
        """
        Play one complete self-play game recording trajectory of leaf states.
        Returns:
          trajectory: list of dicts with 'leaf_phi', 'val', 'turn', 'reward'
          final_outcome: +1.0 (White win), -1.0 (Black win), 0.0 (Draw)
        """
        engine = GameEngine()
        trajectory = []
        ply_count = 0

        while ply_count < max_plies:
            state = engine.get_game_state()
            if state != "ongoing":
                break

            move, leaf = self._select_action_and_leaf(
                engine, depth=self.search_depth, epsilon=self.epsilon
            )
            if move is None:
                break

            # Extract features of the leaf state
            leaf_phi = self.value_fn.extractor.extract_features(leaf)
            leaf_val = self.value_fn.evaluate_features(leaf_phi)

            # Intermediate tactical reward shaping
            tactical_reward = 0.0
            if move.piece_captured != EMPTY_SQUARE:
                captured_val = abs(BASE_PIECE_VALUES.get(move.piece_captured, 0))
                # Reward White for taking Black pieces, reward Black for taking White pieces
                tactical_reward = (captured_val / 1000.0) * (0.05 if engine.white_to_move else -0.05)

            if engine.is_in_check():
                tactical_reward += (0.02 if not engine.white_to_move else -0.02)

            trajectory.append({
                "leaf_phi": leaf_phi,
                "val": leaf_val,
                "turn": engine.white_to_move,
                "reward": tactical_reward,
            })

            engine.make_move(move)
            ply_count += 1

        # Determine terminal outcome
        final_state = engine.get_game_state()
        if final_state == "checkmate":
            # If it's checkmate and white_to_move, Black won. Otherwise White won.
            outcome = -1.0 if engine.white_to_move else 1.0
        else:
            outcome = 0.0  # Stalemate, repetition, 50-move rule, or ply limit

        return trajectory, outcome, ply_count

    def train_episode(self, max_plies=120):
        """
        Execute one TD-Leaf(lambda) self-play episode and update value weights.
        Returns td_loss, outcome, ply_count.
        """
        trajectory, outcome, ply_count = self.play_self_play_game(max_plies=max_plies)
        T = len(trajectory)

        if T == 0:
            return 0.0, outcome, ply_count

        # Compute TD errors and accumulate eligibility traces
        # delta_t = r_{t+1} + gamma * V(s_{t+1}^L) - V(s_t^L)
        # for terminal t=T-1: delta_{T-1} = outcome - V(s_{T-1}^L)
        eligibility = np.zeros(self.value_fn.dim, dtype=np.float64)
        total_td_loss = 0.0

        for t in range(T):
            phi_t = trajectory[t]["leaf_phi"]
            v_t = trajectory[t]["val"]
            reward_t = trajectory[t]["reward"]

            if t == T - 1:
                # Terminal step: target is the true game outcome z in {-1, 0, +1}
                target = outcome
            else:
                target = reward_t + self.gamma * trajectory[t + 1]["val"]

            td_error = target - v_t
            total_td_loss += float(td_error ** 2)

            # Compute analytical gradient: nabla_theta V(s_t^L) = (1 - V_t^2) * phi_t
            grad_t = self.value_fn.get_gradient(phi_t)

            # Update eligibility trace: e_t = gamma * lambda * e_{t-1} + grad_t
            eligibility = self.gamma * self.lam * eligibility + grad_t

            # Clip trace for numerical stability
            np.clip(eligibility, -5.0, 5.0, out=eligibility)

            # Weight update: theta <- theta + alpha * delta_t * e_t
            self.value_fn.update_gradient(eligibility, self.alpha * td_error)

        avg_loss = total_td_loss / max(1, T)

        # Decay epsilon exploration
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return avg_loss, outcome, ply_count

    def train(self, num_episodes=DEFAULT_EPISODES, log_interval=10, snapshot_interval=20):
        """
        Run multi-episode self-play reinforcement learning loop.
        Logs metrics and records weight trajectories.
        """
        print(f"Starting Self-Play TD-Leaf(lambda) Training for {num_episodes} episodes...")
        print(f"Hyperparameters: gamma={self.gamma}, lambda={self.lam}, alpha={self.alpha}, eps={self.epsilon:.3f}")
        start_time = time.time()

        for ep in range(1, num_episodes + 1):
            loss, outcome, plies = self.train_episode()

            # Record stats
            self.history["episodes"].append(ep)
            self.history["td_losses"].append(loss)
            self.history["outcomes"].append(outcome)
            self.history["game_lengths"].append(plies)
            self.history["epsilons"].append(self.epsilon)

            if outcome > 0.5:
                self.history["white_wins"] += 1
            elif outcome < -0.5:
                self.history["black_wins"] += 1
            else:
                self.history["draws"] += 1

            if ep % snapshot_interval == 0 or ep == 1 or ep == num_episodes:
                self.history["weight_history"].append({
                    "episode": ep,
                    "weights": self.value_fn.weights.copy().tolist(),
                    "summary": self.value_fn.get_weights_summary()
                })

            if ep % log_interval == 0 or ep == 1:
                recent_losses = self.history["td_losses"][-log_interval:]
                recent_plies = self.history["game_lengths"][-log_interval:]
                avg_l = sum(recent_losses) / len(recent_losses)
                avg_p = sum(recent_plies) / len(recent_plies)
                w_win_pct = (self.history["white_wins"] / ep) * 100.0
                b_win_pct = (self.history["black_wins"] / ep) * 100.0
                draw_pct = (self.history["draws"] / ep) * 100.0

                elapsed = time.time() - start_time
                print(
                    f"Ep {ep:4d}/{num_episodes} | "
                    f"TD-Loss: {avg_l:.4f} | "
                    f"Avg Plies: {avg_p:4.1f} | "
                    f"W/B/D: {w_win_pct:.1f}%/{b_win_pct:.1f}%/{draw_pct:.1f}% | "
                    f"Eps: {self.epsilon:.3f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.1f}s.")
        return self.history
