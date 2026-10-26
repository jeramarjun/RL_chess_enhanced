# Reinforcement Learning-Enhanced Chess Engine

An 8×8 chess engine project exploring reinforcement learning through self-play and temporal-difference (TD) updates. The learned linear value function is used by a chess player that searches legal moves with iterative-deepening Negamax and alpha-beta pruning.

> **Important implementation note:** the current training code uses epsilon-greedy self-play and a one-ply move/leaf selection routine with TD updates and eligibility traces. This is a TD-learning-style approach; it should not be described as a full multi-ply TD-Leaf(λ) implementation unless the training search is changed to perform the required principal-leaf search.

## Contents

- [Project overview](#project-overview)
- [Reinforcement-learning formulation](#reinforcement-learning-formulation)
- [Repository files](#repository-files)
- [Requirements](#requirements)
- [Installation on Windows](#installation-on-windows)
- [Run a chess match](#run-a-chess-match)
- [Train the agent](#train-the-agent)
- [Evaluate the trained agent](#evaluate-the-trained-agent)
- [Generate plots](#generate-plots)
- [Run tests](#run-tests)
- [Reproduce an experiment](#reproduce-an-experiment)
- [Existing experiment artifacts](#existing-experiment-artifacts)
- [Limitations and interpretation](#limitations-and-interpretation)

## Project overview

The environment is standard chess. The agent uses a 71-dimensional feature vector and a linear value function:

\[
V_\theta(s) = \tanh(\theta^\top \phi(s))
\]

where \(\phi(s)\) is the extracted feature vector and \(\theta\) is the learnable weight vector. The output is bounded between -1 and +1 and represents the position from White's perspective; the player/search code converts it to the side-to-move perspective when needed.

The training script runs self-play episodes. It records leaf-state features and values, computes temporal-difference errors, accumulates eligibility traces, and updates the value-function weights. Training initializes the weights with the project's heuristic-prior initialization; it does not start from all-zero weights by default.

The playing agent includes:
- Iterative-deepening Negamax with alpha-beta pruning.
- Move ordering and a transposition table.
- Quiescence search.
- A per-move time budget and blitz clock tracking.
- A linear learned value function loaded from a JSON weights file.

The project does not require Stockfish or a pretrained chess model.

## Reinforcement-learning formulation

- **State:** a legal chess position, including piece placement, side to move, castling rights, en-passant state, and draw-relevant information handled by the chess environment.
- **Actions:** legal chess moves, including castling, en passant, and pawn promotion.
- **Reward/outcome:** terminal wins, losses, and draws are represented as +1, -1, and 0 from White's perspective. The trainer also computes small auxiliary signals for captures and check-related states.
- **Value function:** a linear combination of extracted features followed by `tanh`.
- **Learning:** temporal-difference updates with eligibility traces, controlled by the learning rate (`alpha`), discount factor (`gamma`), trace-decay parameter (`lam`), and epsilon-greedy exploration.

The feature extractor includes material, piece-square, pawn-structure, mobility/center, king-safety, development, and bishop-pair features. The exact feature names and ordering are stored in `trained_weights.json`; the feature dimension is 71.

## Repository files

The project files are organised at the repository root:

| File | Purpose |
|---|---|
| `board.py` | Chess board wrapper and game-state handling using `python-chess`. |
| `config.py` | Board constants, piece values, and default hyperparameters. |
| `features.py` | Extracts the 71-dimensional state feature vector. |
| `rl_value_net.py` | Linear value function, gradient, weight persistence, and a neural-network class. |
| `td_learning.py` | Self-play trainer and TD updates with eligibility traces. |
| `rl_player.py` | RL-enhanced chess player and search. |
| `baselines.py` | Random, greedy-material, untrained, and heuristic opponents. |
| `train.py` | Training command-line entry point. |
| `evaluate.py` | Benchmark evaluation against the included baselines. |
| `game_runner.py` | Runs a chess match and displays the board. |
| `plot_metrics.py` | Generates plots from saved training/evaluation JSON files. |
| `test_suite.py` | Unit tests. |
| `trained_weights.json` | Saved linear value-function weights. |
| `training_metrics.json` | Saved training history. |
| `evaluation_results.json` | Saved baseline benchmark results. |
| `requirements.txt` | Python dependencies. |
| `RL_Chess_Engine_Report.pdf` | Existing project report artifact, if included in the checkout. |
| `chess_rl_agent.py`, `b24cm1034.py` | Agent/interface wrappers. |

Training also creates a `checkpoints/` directory and periodic checkpoint JSON files. `plot_metrics.py` creates a `figures/` directory when it runs.

## Requirements

- Python 3.10 or newer is recommended.
- Git, if cloning the repository.
- Packages listed in `requirements.txt`:
  - `chess` (`python-chess`)
  - `numpy`
  - `matplotlib`
  - `torch`
  - `reportlab`

The chess engine and linear TD training can run on a CPU. A dedicated GPU is not required for the documented linear training workflow.

## Installation on Windows

Open PowerShell in the cloned repository folder.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the virtual environment's Python executable directly in the commands below, or change the local execution policy only if you understand and are permitted to do so.

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Check that the core dependencies import

```powershell
python -c "import chess, numpy, matplotlib, torch; print('Core dependencies imported successfully')"
```

If this command fails, resolve the reported dependency/import error before running the scripts.

## Run a chess match

Run the RL-enhanced player as White against the heuristic baseline as Black:

```powershell
python game_runner.py --white rl --black heuristic
```

Run against a random opponent:

```powershell
python game_runner.py --white rl --black random
```

Supported player names are `rl`, `random`, `greedy`, `untrained`, and `heuristic`.

Useful options:

```powershell
python game_runner.py --white rl --black heuristic --time 60 --turns 150
python game_runner.py --white rl --black random --quiet
```

- `--time` sets the total blitz clock in seconds.
- `--turns` sets the maximum number of plies before the runner stops the game.
- `--quiet` suppresses per-turn board printing.

The RL player loads `trained_weights.json` by default when that file exists. A draw is a valid chess outcome and does not by itself indicate a training or environment failure; inspect the reported game-ending condition to distinguish repetition, stalemate, rule-based draws, and a configured ply limit.

## Train the agent

A short smoke run is a good first check:

```powershell
python train.py --episodes 2 --max-plies 40 --log-interval 1 --output-weights runs/smoke_weights.json --output-metrics runs/smoke_metrics.json --checkpoint-dir runs/smoke_checkpoints
```

This writes new output files under `runs/` and avoids overwriting the root-level saved artifacts.

Run a longer training experiment with the documented hyperparameters:

```powershell
python train.py --episodes 100 --max-plies 80 --alpha 0.002 --gamma 0.99 --lam 0.70 --epsilon 0.15 --epsilon-decay 0.995 --min-epsilon 0.02 --log-interval 10 --output-weights runs/weights_100.json --output-metrics runs/metrics_100.json --checkpoint-dir runs/checkpoints_100
```

The actual epsilon-decay default is defined in `config.py`; set `--epsilon-decay` explicitly when reproducing a run.

### Training arguments

| Argument | Meaning |
|---|---|
| `--episodes` | Number of self-play training episodes. |
| `--alpha` | TD learning rate. |
| `--gamma` | Discount factor. |
| `--lam` | Eligibility-trace decay parameter. |
| `--epsilon` | Initial epsilon-greedy exploration rate. |
| `--epsilon-decay` | Exploration decay multiplier. |
| `--min-epsilon` | Lower bound on exploration. |
| `--max-plies` | Maximum plies per training game; a cutoff is treated as a draw by the current trainer. |
| `--log-interval` | How often training progress is printed. |
| `--output-weights` | Destination for final weights JSON. |
| `--output-metrics` | Destination for training metrics JSON. |
| `--checkpoint-dir` | Directory for periodic checkpoint weights. |

**Resume behavior:** the current `train.py` initializes a new value function at the start of each run and does not expose a resume-from-checkpoint argument. Do not use it as a resume command; implement and test checkpoint restoration before claiming resume support.

## Evaluate the trained agent

Run a quick benchmark with two games per baseline and a short ply cap:

```powershell
python evaluate.py --weights trained_weights.json --games-per-baseline 2 --max-plies 40 --output runs/evaluation_smoke.json
```

Run a larger benchmark:

```powershell
python evaluate.py --weights trained_weights.json --games-per-baseline 10 --max-plies 80 --output runs/evaluation_10_per_baseline.json
```

The evaluation script compares the loaded RL player against the included random, greedy-material, untrained-RL, and heuristic baselines. It writes the results to the path supplied by `--output`.

### Evaluation arguments

| Argument | Meaning |
|---|---|
| `--weights` | Path to the trained weights JSON file. |
| `--games-per-baseline` | Number of games played against each baseline. |
| `--max-plies` | Maximum plies per benchmark game. |
| `--output` | Path for the output JSON file. |

The script uses its own configured search depth/time budget for the RL player. The untrained and heuristic opponents have their own settings, so the saved benchmark should be described as a comparison against these implemented baselines, not as a perfectly controlled ablation unless their search settings are explicitly matched.

For a fairer comparison, use an even number of games per opponent so both colors are represented equally. Report the number of games and treat small tournaments as preliminary evidence rather than a reliable Elo rating.

## Generate plots

With `training_metrics.json` and `evaluation_results.json` present in the repository root, run:

```powershell
python plot_metrics.py
```

The script creates the `figures/` directory and generates the training curves, weight trajectories, baseline comparison, and search-efficiency plot files. Check the script output and confirm the generated files exist before referring to them in a report.

To plot a new run, place or copy its metrics/results at the paths the script reads, or update the script to accept explicit paths. Preserve original JSON artifacts before replacing them.

## Run tests

The uploaded project test file is at the repository root as `test_suite.py`. Run it with:

```powershell
python -m unittest test_suite -v
```

Alternatively:

```powershell
python test_suite.py
```

Run tests after installing the dependencies. A passing test suite checks only the covered behaviors; it does not by itself prove that the RL agent is strong or that every search edge case is correct.

## Reproduce an experiment

For a reproducible run:

1. Use the same Python version and install `requirements.txt`.
2. Keep a copy of the existing weights, metrics, and evaluation JSON files.
3. Record the exact command-line arguments and random seeds, if seed support is added to the scripts.
4. Train into new output paths rather than overwriting the existing artifacts.
5. Evaluate with an even number of games per baseline and a documented ply limit.
6. Keep the resulting JSON files and use them to generate plots.
7. Include the game counts, color assignment, hyperparameters, and limitations in the report.

The current scripts do not expose a unified experiment seed option. Results may vary because of random move selection and exploration; for strict reproducibility, add explicit seed handling to the code and record the seed with each run.

## Existing experiment artifacts

The repository's saved `evaluation_results.json` records eight games against each of four baselines (32 games total). These are existing artifacts, not a promise that a fresh run will reproduce the same outcomes.

| Baseline | Games | Wins | Draws | Losses | Score |
|---|---:|---:|---:|---:|---:|
| Random | 8 | 2 | 6 | 0 | 62.5% |
| Greedy material | 8 | 7 | 1 | 0 | 93.75% |
| Untrained RL | 8 | 6 | 2 | 0 | 87.5% |
| Heuristic alpha-beta | 8 | 1 | 7 | 0 | 56.25% |

Score is calculated as `(wins + 0.5 × draws) / games`. These samples are small, so use them as preliminary results. The existing metrics also contain many games stopped at the training ply limit; the current trainer records such games as draws. Separate those truncations from genuine chess draws when interpreting training performance.

The existing `training_metrics.json` contains 100 episodes. Its TD losses fluctuate; do not describe the loss as monotonically decreasing. The weights file records the actual saved feature weights. When updating the report, use the JSON artifacts and newly reproduced experiments rather than manually adjusting claims to match an earlier narrative.

## Limitations and interpretation

- The current training move/leaf selection is shallow (one ply), so the training method is not a full multi-ply TD-Leaf(λ) search as described in the original project proposal.
- The trainer treats games that reach `--max-plies` as draws. This can inflate the draw count compared with games ending under official chess rules.
- The current evaluation sample is small and the opponents do not all use identical search settings. It does not establish a reliable Elo rating or prove that all strength differences come from learning.
- Reward shaping in the implementation should be described according to the code; do not claim a formal potential-based policy-invariance guarantee unless the implemented shaping function satisfies that formulation.
- Search correctness, especially quiescence handling while in check and repetition-sensitive transposition-table reuse, should be covered by regression tests before making strong claims about tactical reliability.
- The current training script starts from its configured heuristic-prior initialization and has no resume flag.

## Assignment/report checklist

Before submission, verify that the README commands run in a clean virtual environment, the test suite results are recorded honestly, the final weights are loadable, and the report's figures and benchmark table match the saved JSON data. State the implemented TD method accurately and distinguish measured results from goals or estimates.
