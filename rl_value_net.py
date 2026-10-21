"""
Value Function representations for Reinforcement Learning in Chess.
Provides:
  1. LinearValueFunction with analytical gradient and tanh output activation.
  2. NeuralValueNetwork (PyTorch MLP) with backpropagation.
"""

import json
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
from features import FeatureExtractor, FEATURE_DIM


class LinearValueFunction:
    """
    Parametric Linear Value Function:
      V_theta(s) = tanh( theta^T phi(s) )
    The tanh squashes the evaluation into (-1, 1), naturally aligning with
    chess game outcomes (+1 win, -1 loss, 0 draw).

    Analytical Gradient:
      nabla_theta V(s) = (1 - V(s)^2) * phi(s)
    """

    def __init__(self, feature_dim=FEATURE_DIM, init_strategy="heuristic_prior"):
        self.dim = feature_dim
        self.weights = np.zeros(self.dim, dtype=np.float64)
        self.extractor = FeatureExtractor()

        if init_strategy == "heuristic_prior":
            self.init_heuristic_prior()
        elif init_strategy == "random":
            self.weights = np.random.normal(0.0, 0.05, self.dim)
        elif init_strategy == "zeros":
            self.weights = np.zeros(self.dim, dtype=np.float64)

    def init_heuristic_prior(self):
        """
        Initialize weights with standard baseline chess heuristics as a starting prior.
        RL self-play will subsequently adapt, optimize, and fine-tune these weights.
        """
        # Material weights (mapped to normalized features)
        # Pawns: 1.0, Knights: 3.2, Bishops: 3.3, Rooks: 5.0, Queens: 9.0
        names = self.extractor.feature_names
        name_to_idx = {name: i for i, name in enumerate(names)}

        # Net material
        if "net_pawns" in name_to_idx:
            self.weights[name_to_idx["net_pawns"]] = 0.80
        if "net_knights" in name_to_idx:
            self.weights[name_to_idx["net_knights"]] = 2.40
        if "net_bishops" in name_to_idx:
            self.weights[name_to_idx["net_bishops"]] = 2.50
        if "net_rooks" in name_to_idx:
            self.weights[name_to_idx["net_rooks"]] = 3.80
        if "net_queens" in name_to_idx:
            self.weights[name_to_idx["net_queens"]] = 7.20

        # PST weights
        pst_features = [
            "net_pst_pawns", "net_pst_knights", "net_pst_bishops",
            "net_pst_rooks", "net_pst_queens", "net_pst_kings"
        ]
        for f in pst_features:
            if f in name_to_idx:
                self.weights[name_to_idx[f]] = 0.25

        # Tactical and structural features
        if "net_passed_pawns" in name_to_idx:
            self.weights[name_to_idx["net_passed_pawns"]] = 0.35
        if "net_doubled_pawns" in name_to_idx:
            self.weights[name_to_idx["net_doubled_pawns"]] = 0.15
        if "net_isolated_pawns" in name_to_idx:
            self.weights[name_to_idx["net_isolated_pawns"]] = 0.15
        if "net_connected_pawns" in name_to_idx:
            self.weights[name_to_idx["net_connected_pawns"]] = 0.20
        if "net_center_pawns" in name_to_idx:
            self.weights[name_to_idx["net_center_pawns"]] = 0.30

        # Mobility and center
        if "net_mobility" in name_to_idx:
            self.weights[name_to_idx["net_mobility"]] = 0.20
        if "net_center_control" in name_to_idx:
            self.weights[name_to_idx["net_center_control"]] = 0.25

        # King safety
        if "in_check_w" in name_to_idx:
            self.weights[name_to_idx["in_check_w"]] = -0.40
        if "in_check_b" in name_to_idx:
            self.weights[name_to_idx["in_check_b"]] = 0.40
        if "net_king_shield" in name_to_idx:
            self.weights[name_to_idx["net_king_shield"]] = 0.25

        # Castling & development
        if "net_undeveloped_minors" in name_to_idx:
            self.weights[name_to_idx["net_undeveloped_minors"]] = 0.20
        if "net_bishop_pair" in name_to_idx:
            self.weights[name_to_idx["net_bishop_pair"]] = 0.25

    def evaluate_features(self, phi):
        """Compute V_theta(phi) in (-1, 1) from White's perspective."""
        raw = np.dot(self.weights, phi)
        # Numerical stability clip
        raw_clipped = np.clip(raw, -15.0, 15.0)
        return float(np.tanh(raw_clipped))

    def evaluate_state(self, engine):
        """Extract features and compute V_theta(s)."""
        phi = self.extractor.extract_features(engine)
        return self.evaluate_features(phi)

    def evaluate_side(self, engine):
        """
        Compute evaluation from the perspective of the current side to move.
        Returns positive if current player has advantage, negative otherwise.
        """
        v_white = self.evaluate_state(engine)
        return v_white if engine.white_to_move else -v_white

    def to_centipawns(self, v):
        """Convert bounded (-1, 1) RL value into chess centipawn equivalent."""
        # Using inverse tanh / scaled tanh mapping
        # 0.0 -> 0 cp, 0.5 -> ~200 cp, 0.9 -> ~700 cp
        v_clip = np.clip(v, -0.999, 0.999)
        return 400.0 * np.arctanh(v_clip)

    def get_gradient(self, phi):
        """
        Compute analytical gradient nabla_theta V(s):
          nabla_theta V(s) = (1 - V^2) * phi
        """
        v = self.evaluate_features(phi)
        scale = 1.0 - (v * v)
        return scale * np.array(phi, dtype=np.float64)

    def update_gradient(self, grad, step_size):
        """Update weights: theta <- theta + step_size * grad."""
        self.weights += step_size * grad

    def save_weights(self, filepath):
        """Save weights to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "feature_dim": self.dim,
            "feature_names": self.extractor.feature_names,
            "weights": self.weights.tolist(),
        }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

    def load_weights(self, filepath):
        """Load weights from JSON file."""
        with open(filepath, "r") as f:
            payload = json.load(f)
        self.weights = np.array(payload["weights"], dtype=np.float64)
        if len(self.weights) != self.dim:
            raise ValueError(f"Weight dimension mismatch: expected {self.dim}, got {len(self.weights)}")

    def get_weights_summary(self):
        """Return dict of named weights for inspection and logging."""
        return {
            name: float(self.weights[i])
            for i, name in enumerate(self.extractor.feature_names)
        }


class NeuralValueNetwork(nn.Module):
    """
    PyTorch Deep Value Network for Chess:
      R^FEATURE_DIM -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(1) -> Tanh
    """

    def __init__(self, input_dim=FEATURE_DIM, hidden_dim1=128, hidden_dim2=64):
        super(NeuralValueNetwork, self).__init__()
        self.extractor = FeatureExtractor()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def evaluate_state(self, engine):
        phi = self.extractor.extract_features(engine)
        tensor_x = torch.tensor(phi, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            val = self.forward(tensor_x).item()
        return val

    def evaluate_side(self, engine):
        v_white = self.evaluate_state(engine)
        return v_white if engine.white_to_move else -v_white

    def save_model(self, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=torch.device("cpu")))
        self.eval()
