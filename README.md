# Reinforcement Learning-Enhanced Chess Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Reinforcement Learning](https://img.shields.io/badge/Algorithm-TD--Leaf(%CE%BB)%20Self--Play-orange.svg)]()
[![Search](https://img.shields.io/badge/Search-Alpha--Beta%20Pruning%20%2B%20Quiescence-green.svg)]()
[![Course](https://img.shields.io/badge/IIT%20Jodhpur-AI%20Assignment%202%20(Mini--Project)-red.svg)]()

> **Autonomous 8×8 Chess Strategic Decision-Making via Temporal-Difference Learning, Self-Play Value Optimization, and Minimax Search Integration.**

---

## 📌 Project Overview

Traditional computer chess systems rely heavily on manually engineered, static heuristic evaluation functions and supervised opening books. While effective, heuristic functions inject designer bias, require decades of trial-and-error hand tuning, and frequently fail in non-standard tactical or structural positions.

**This project formulates standard 8×8 chess strictly as a Reinforcement Learning (RL) problem.** The agent autonomously learns strategic evaluation weights from scratch through self-play and **Temporal-Difference Learning (TD-Leaf($\lambda$))**.

The learned evaluation function is integrated into a high-performance **Iterative-Deepening Negamax search engine with $\alpha$-$\beta$ pruning, quiescence search, transposition tables, and blitz clock management (60-second shared clock)**.

---

## 🧠 Reinforcement Learning Formulation

In compliance with the assignment specification, the environment and agent are formulated as a Markov Decision Process (MDP): $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$.

### 1. State Space ($\mathcal{S}$)
A chess state $s \in \mathcal{S}$ is formally represented by the tuple:
$$s = \langle \mathbf{B}, t, \mathbf{c}, ep, h_{50}, h_{\text{rep}} \rangle$$
- $\mathbf{B} \in \{0, 1\}^{12 \times 8 \times 8}$: Spatial occupancy tensor encoding the positions of the 6 piece types ($\{P, N, B, R, Q, K\}$) for both colors across 64 board squares.
- $t \in \{0, 1\}$: Active player turn (White or Black).
- $\mathbf{c} \in \{0, 1\}^4$: Castling rights (kingside/queenside for White and Black).
- $ep \in \{0, \dots, 63\} \cup \{\emptyset\}$: En-passant target square availability.
- $h_{50} \in [0, 100]$: Half-move clock for the 50-move draw rule.
- $h_{\text{rep}}$: Position repetition history for threefold repetition detection.

To facilitate sample-efficient TD gradient propagation, each state is mapped to a normalized 71-dimensional strategic feature vector $\boldsymbol{\phi}(s) \in \mathbb{R}^{71}$:
$$\boldsymbol{\phi}(s) = \left[ \boldsymbol{\phi}_{\text{mat}}, \boldsymbol{\phi}_{\text{pst}}, \boldsymbol{\phi}_{\text{pawn}}, \boldsymbol{\phi}_{\text{mob}}, \boldsymbol{\phi}_{\text{king}}, \boldsymbol{\phi}_{\text{dev}}, 1.0 \right]^T$$

| Feature Category | Dimension | Strategic Description |
| :--- | :---: | :--- |
| **Material Balance** | 16 | Piece counts ($P, N, B, R, Q$), net differentials, and non-pawn endgame phase scaling. |
| **Piece-Square Tables (PST)** | 18 | Centralization and square-occupancy metrics per piece type for White, Black, and Net. |
| **Pawn Structure** | 15 | Passed pawns, doubled pawns, isolated pawns, connected phalanxes, and center occupation ($d4/e4/d5/e5$). |
| **Mobility & Activity** | 6 | Attack bitboard density across minor/major pieces and key center square pressure. |
| **King Safety & Checks** | 7 | Direct check status, castled pawn shields, and enemy major-piece tropism (proximity). |
| **Development & Harmony** | 8 | Castling execution, undeveloped back-rank minors, and bishop pair advantage. |
| **Constant Bias** | 1 | Baseline offset term for balance calibration. |

### 2. Action Space ($\mathcal{A}$)
The action space $\mathcal{A}(s)$ is discrete and context-dependent, comprising all legal chess moves $a \in \mathcal{M}(s)$ conforming to standard FIDE rules (including castling, en passant, and pawn promotions to $\{Q, R, B, N\}$). The branching factor satisfies $|\mathcal{A}(s)| \in [1, 218]$ with an average branching factor $\bar{b} \approx 35$.

### 3. Reward Function ($\mathcal{R}$) & Incentive Design
Terminal rewards reflect true game outcomes from White's perspective:
$$R_{\text{terminal}}(s) = \begin{cases} +1.0 & \text{if White wins (checkmate)}, \\ -1.0 & \text{if Black wins (checkmate)}, \\ 0.0 & \text{if Draw (stalemate, 50-move rule, threefold repetition, insufficient material)}. \end{cases}$$

To provide intermediate feedback during self-play, the implementation uses tactical reward shaping. Captures receive a bonus proportional to the captured piece's value, and check-related events receive a small bonus or penalty depending on the player. The final game outcome is used as the terminal learning target: +1 for a White win, -1 for a Black win, and 0 for a draw. These intermediate rewards are direct tactical bonuses; the implementation does not implement the formal potential-based reward-shaping method.
$$r_t = R_{\text{terminal}}(s_t) + c_{\text{mat}} \cdot \frac{\Delta \text{Material}}{1000} + c_{\text{check}} \cdot \mathbb{I}(\text{check})$$
where $c_{\text{mat}} = 0.05$ and $c_{\text{check}} = 0.02$. Potential-based shaping guarantees optimal policy invariance while accelerating TD error convergence.

### 4. Value Function Parameterization
The value function estimates the expected discounted return from state $s$:
$$V_{\boldsymbol{\theta}}(s) = \tanh\left( \boldsymbol{\theta}^T \boldsymbol{\phi}(s) \right) \in (-1, 1)$$
The hyperbolic tangent activation naturally bounds evaluations to the game outcome range $(-1, 1)$, preventing exploding gradient estimates and yielding the analytical gradient:
$$\nabla_{\boldsymbol{\theta}} V_{\boldsymbol{\theta}}(s) = \left( 1 - V_{\boldsymbol{\theta}}(s)^2 \right) \cdot \boldsymbol{\phi}(s)$$

---

## ⚡ Reinforcement Learning Algorithm: TD-Leaf($\lambda$)

Standard TD($\lambda$) evaluates root states directly. However, in adversarial search games, an agent selects moves based on deep minimax projections. **TD-Leaf($\lambda$)** (Baxter, Tridgell, Weaver, 1998) bridges this gap by defining the evaluation of root state $s_t$ as the value of the principal leaf state $s_t^L = \text{Leaf}(s_t, d)$ discovered by minimax search:
$$J(s_t) = V_{\boldsymbol{\theta}}(s_t^L)$$

### Self-Play Learning Cycle
1. **Self-Play Move Selection**: The agent selects moves via search with $\epsilon$-greedy exploration ($\epsilon$ annealed from $0.15 \to 0.02$).
2. **Temporal Difference Error**:
   $$\delta_t = \begin{cases} r_{t+1} + \gamma J(s_{t+1}) - J(s_t) & \text{for non-terminal plies } t < T - 1 \\ z - J(s_T) & \text{for terminal ply } t = T - 1 \end{cases}$$
   where $z \in \{+1.0, -1.0, 0.0\}$ is the final game outcome.
3. **Eligibility Traces**:
   $$\mathbf{e}_t = \gamma \lambda \mathbf{e}_{t-1} + \nabla_{\boldsymbol{\theta}} J(s_t)$$
4. **Parameter Updates**:
   $$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \delta_t \mathbf{e}_t$$
   with learning rate $\alpha = 0.002$, discount factor $\gamma = 0.99$, and trace decay $\lambda = 0.70$.

---

## 🔍 Alpha-Beta Search Integration

The learned RL value function is integrated into an iterative-deepening search engine optimized for tournament blitz play:
- **Iterative Deepening Negamax**: Searches progressively deeper plies ($d = 1, 2, 3, \dots$) within allocated move time slices.
- **Quiescence Search**: Extends search on tactical capture sequences at leaf nodes to eliminate the horizon effect.
- **Transposition Table**: Caches exact, lower, and upper search bounds indexed by 64-bit Zobrist position hashes.
- **Move Ordering**: Prioritizes Transposition Table principal variation (PV) moves, MVV-LVA (Most Valuable Victim - Least Valuable Attacker) captures, killer moves, and history scores.
- **Blitz Clock Allocation**: Manages a 60-second total game clock with dynamic time budgeting per phase and an emergency safety reserve (2.0s).

---

## 📊 Systematic Benchmark Comparisons

The trained RL-Enhanced engine was evaluated against four distinct baseline architectures across alternating-color blitz tournaments (60s clock):

| Opponent Baseline | Type / Role | Win Rate | Draw Rate | Loss Rate | Match Score | $\Delta$Elo |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Player** | Uniform random legal play | **25.0%** | 75.0% | **0.0%** | 0.625 | **+88.7** |
| **Greedy Material** | 1-ply immediate capture maximization | **87.5%** | 12.5% | **0.0%** | 0.938 | **+470.4** |
| **Untrained RL ($\boldsymbol{\theta}_0$)** | Same search engine with unlearned weights (Ablation) | **75.0%** | 25.0% | **0.0%** | 0.875 | **+338.0** |
| **Heuristic Minimax** | Classical Alpha-Beta with fixed handcrafted PSTs | **12.5%** | 87.5% | **0.0%** | 0.562 | **+43.7** |

### Key Experimental Insights
1. **TD Loss Convergence**: Mean squared TD error decreased monotonically from $0.153$ to $<0.032$, indicating stable value approximation.
2. **Autonomous Strategic Discovery**: Without human guidance, the agent learned to elevate pawn structure bonuses ($+0.35 \to +0.48$) and center control ($+0.25 \to +0.39$), prioritizing piece coordination over naive material preservation.
3. **Ablation Validation**: The 75.0% win rate (and 0 losses) against the identical search architecture operating on unlearned weights ($\boldsymbol{\theta}_0$) proves that strategic strength stems directly from Reinforcement Learning.

---

## 📁 Repository Structure

```
Chess_RL/
├── src/                          # Core engine & learning modules
│   ├── __init__.py               # Source package initializer
│   ├── config.py                 # Board constants, piece values, PSTs, hyperparameters
│   ├── board.py                  # Standard 8x8 Chess rules wrapper using python-chess
│   ├── features.py               # 71-dimensional state feature extractor phi(s)
│   ├── rl_value_net.py           # Linear value function used by the TD training pipeline
│   ├── td_learning.py            # TD-Leaf(lambda) self-play training algorithm
│   ├── rl_player.py              # Iterative-deepening Alpha-Beta search player
│   ├── baselines.py              # Random, Greedy, Untrained, and Heuristic players
│   └── game_runner.py            # Blitz match runner with Unicode board display
├── models/                       # Trained model weights and checkpoints
│   ├── trained_weights.json      # Final learned evaluation weights
│   ├── training_metrics.json     # Saved training loss, plies, and weight history
│   └── checkpoints/              # Multi-episode snapshot weights
├── data/                         # Benchmark & tournament data
│   └── evaluation_results.json   # Systematic tournament results vs all baselines
├── figures/                      # High-resolution publication plots
│   ├── training_curves.png
│   ├── weight_trajectories.png
│   ├── baseline_comparison.png
│   └── search_efficiency.png
├── report/                       # 2-Page Descriptive Report deliverables
│   ├── RL_Chess_Engine_Report.pdf # Final compiled 2-page report PDF
│   ├── report.tex                # IEEE two-column LaTeX source code
│   └── generate_report_pdf.py    # ReportLab PDF generator script
├── tests/                        # Automated unit tests
│   ├── __init__.py
│   └── test_suite.py             # 15-test unit test suite
├── train.py                      # Root entry point: Model training CLI
├── evaluate.py                   # Root entry point: Baseline benchmarking CLI
├── plot_metrics.py               # Root entry point: Visualization generator
├── game_runner.py                # Root entry point: Interactive blitz match runner
├── chess_rl_agent.py             # Course submission wrapper (tournament import)
├── b24cm1034.py                  # Roll number agent wrapper alias
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # Comprehensive documentation & reproduction guide
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Installation
Ensure Python 3.10+ is installed, then install all dependencies:
```bash
python -m pip install -r requirements.txt
```

### 2. Run a Blitz Match
Watch the RL-Enhanced agent play against a baseline opponent on a 60-second Blitz clock:
```bash
# RL Agent (White) vs Classical Heuristic Minimax (Black)
python game_runner.py --white rl --black heuristic

# RL Agent (White) vs Random Agent (Black)
python game_runner.py --white rl --black random
```

### 3. Run Self-Play RL Training
Train the evaluation weights from scratch using TD-Leaf($\lambda$) self-play:
```bash
python train.py --episodes 100 --max-plies 80 --alpha 0.002 --gamma 0.99 --lam 0.70
```
This produces `trained_weights.json`, `training_metrics.json`, and intermediate snapshots in `checkpoints/`.

### 4. Run Benchmark Tournament Evaluations
Execute head-to-head match series against all baseline players:
```bash
python evaluate.py --weights trained_weights.json --games-per-baseline 8 --max-plies 60
```
Results are saved to `evaluation_results.json`.

### 5. Generate Publication Graphs
Generate all report figures:
```bash
python plot_metrics.py
```
Outputs high-resolution charts in `figures/`.

### 6. Run Unit Tests
Verify board mechanics, feature extraction, TD updates, and Alpha-Beta search soundness:
```bash
python -m unittest discover tests
```



## 👥 Authors & Academic Metadata

- **Course:** AI Course: Assignment 2 &mdash; Real-Life Use Cases of Reinforcement Learning (Mini-Project)
- **Institution:** Indian Institute of Technology Jodhpur (IITJ)
- **Team Submission Roll Numbers:** B24CM1034, B24CM1037


---

## 📜 References
1. Baxter, J., Tridgell, A., & Weaver, L. (1998). *TDLeaf($\lambda$): Combining Temporal Difference Learning with Game-Tree Search.* Australian Joint Conference on Artificial Intelligence.
2. Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.* ICML.
3. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction.* 2nd Edition, MIT Press.
4. Silver, D., et al. (2018). *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.* Science, 362(6419), 1140-1144.
