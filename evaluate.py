"""
Systematic Performance Evaluation & Baseline Benchmark Suite.
Compares the RL-Enhanced Chess Engine against:
  1. RandomPlayer (P25CS0004 / P22CS201 baseline)
  2. UntrainedRLPlayer (Search with zero/random weights theta_0)
  3. HeuristicPlayer (Classical non-learning minimax search with PST tables)
  4. GreedyMaterialPlayer (Greedy 1-ply tactical baseline)
Computes win/draw/loss rates, average game length, search node counts,
and empirical Elo differentials.
"""

import argparse
import json
import math
import os
import sys
import time

_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_current_dir, "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from board import GameEngine, Move
from config import EMPTY_SQUARE, BASE_PIECE_VALUES
from rl_player import RLEnhancedPlayer
from baselines import RandomPlayer, UntrainedRLPlayer, HeuristicPlayer, GreedyMaterialPlayer


def play_match(white_agent_cls, black_agent_cls, white_kwargs=None, black_kwargs=None, max_plies=120):
    """
    Play a single head-to-head game between two agent configurations.
    Returns:
      result: 'white_win', 'black_win', 'draw'
      plies: int
      white_nodes: int (total nodes expanded by White)
      black_nodes: int (total nodes expanded by Black)
      white_time: float (total time used by White)
      black_time: float (total time used by Black)
      material_diff: int (White material - Black material at game end)
    """
    white_kwargs = white_kwargs or {}
    black_kwargs = black_kwargs or {}

    engine = GameEngine()
    white_player = white_agent_cls(engine, **white_kwargs)
    black_player = black_agent_cls(engine, **black_kwargs)

    plies = 0
    total_w_nodes = 0
    total_b_nodes = 0
    total_w_time = 0.0
    total_b_time = 0.0

    while plies < max_plies:
        state = engine.get_game_state()
        if state != "ongoing":
            break

        player = white_player if engine.white_to_move else black_player
        is_white = engine.white_to_move

        t0 = time.perf_counter()
        move = player.get_best_move()
        dt = time.perf_counter() - t0

        if is_white:
            total_w_time += dt
            total_w_nodes += getattr(player, "nodes_expanded", 1)
        else:
            total_b_time += dt
            total_b_nodes += getattr(player, "nodes_expanded", 1)

        if not move:
            break

        engine.make_move(move)
        plies += 1

    final_state = engine.get_game_state()
    if final_state == "checkmate":
        result = "black_win" if engine.white_to_move else "white_win"
    else:
        result = "draw"

    # Compute final material differential
    w_mat = 0
    b_mat = 0
    for r in range(8):
        for c in range(8):
            p = engine.board[r][c]
            if p != EMPTY_SQUARE:
                v = abs(BASE_PIECE_VALUES.get(p, 0))
                if p[0] == "w":
                    w_mat += v
                else:
                    b_mat += v
    mat_diff = w_mat - b_mat

    return {
        "result": result,
        "plies": plies,
        "white_nodes": total_w_nodes,
        "black_nodes": total_b_nodes,
        "white_time": total_w_time,
        "black_time": total_b_time,
        "material_diff": mat_diff,
    }


def run_benchmark_series(agent_a_name, agent_a_cls, agent_a_kwargs,
                         agent_b_name, agent_b_cls, agent_b_kwargs,
                         num_games=10, max_plies=100):
    """
    Run an alternating-color match series between Agent A and Agent B.
    Plays num_games / 2 with A as White and num_games / 2 with A as Black.
    """
    print(f"\n--- Benchmark Series: {agent_a_name} vs {agent_b_name} ({num_games} games) ---")
    
    a_wins = 0
    b_wins = 0
    draws = 0
    total_plies = 0
    total_a_nodes = 0
    total_a_time = 0.0

    games_per_side = num_games // 2
    rem = num_games % 2

    # Agent A as White
    for g in range(games_per_side + rem):
        res = play_match(
            agent_a_cls, agent_b_cls,
            white_kwargs=agent_a_kwargs, black_kwargs=agent_b_kwargs,
            max_plies=max_plies
        )
        total_plies += res["plies"]
        total_a_nodes += res["white_nodes"]
        total_a_time += res["white_time"]
        if res["result"] == "white_win":
            a_wins += 1
        elif res["result"] == "black_win":
            b_wins += 1
        else:
            draws += 1

    # Agent A as Black
    for g in range(games_per_side):
        res = play_match(
            agent_b_cls, agent_a_cls,
            white_kwargs=agent_b_kwargs, black_kwargs=agent_a_kwargs,
            max_plies=max_plies
        )
        total_plies += res["plies"]
        total_a_nodes += res["black_nodes"]
        total_a_time += res["black_time"]
        if res["result"] == "black_win":
            a_wins += 1
        elif res["result"] == "white_win":
            b_wins += 1
        else:
            draws += 1

    score = (a_wins + 0.5 * draws) / num_games
    win_rate = (a_wins / num_games) * 100.0
    draw_rate = (draws / num_games) * 100.0
    loss_rate = (b_wins / num_games) * 100.0
    avg_plies = total_plies / num_games
    avg_nodes_per_move = total_a_nodes / max(1, total_plies // 2)
    avg_time_per_move = total_a_time / max(1, total_plies // 2)

    # Estimate Elo difference: delta_elo = -400 * log10(1/score - 1)
    if score >= 0.999:
        elo_diff = +800.0
    elif score <= 0.001:
        elo_diff = -800.0
    else:
        elo_diff = round(-400.0 * math.log10((1.0 / score) - 1.0), 1)

    print(f"Results: {agent_a_name} Wins: {a_wins} ({win_rate:.1f}%) | "
          f"Draws: {draws} ({draw_rate:.1f}%) | "
          f"Losses: {b_wins} ({loss_rate:.1f}%) | Score: {score:.3f} | "
          f"Estimated dElo: {elo_diff:+.1f}")
    print(f"Stats: Avg Plies: {avg_plies:.1f} | Avg Nodes/Move: {avg_nodes_per_move:.1f} | Avg Time/Move: {avg_time_per_move*1000:.1f}ms")

    return {
        "opponent": agent_b_name,
        "games": num_games,
        "wins": a_wins,
        "draws": draws,
        "losses": b_wins,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "score": score,
        "elo_diff": elo_diff,
        "avg_plies": avg_plies,
        "avg_nodes_per_move": avg_nodes_per_move,
        "avg_time_per_move_ms": avg_time_per_move * 1000.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Systematic Evaluation Benchmark for RL Chess Engine")
    parser.add_argument("--weights", type=str, default="trained_weights.json", help="Path to trained weights JSON")
    parser.add_argument("--games-per-baseline", type=int, default=10, help="Number of games per baseline opponent")
    parser.add_argument("--max-plies", type=int, default=80, help="Max plies per benchmark game")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Path to save results JSON")
    args = parser.parse_args()

    print("=" * 65)
    print("  SYSTEMATIC PERFORMANCE EVALUATION: RL AGENT VS BASELINES  ")
    print("=" * 65)
    print(f"Trained Weights:     {args.weights}")
    print(f"Games Per Baseline:  {args.games_per_baseline}")
    print(f"Max Plies:           {args.max_plies}")
    print("=" * 65)

    rl_kwargs = {"weights_path": args.weights, "max_depth": 3, "time_limit": 0.25}

    results = {}

    # 1. vs Random Baseline
    res_rand = run_benchmark_series(
        agent_a_name="RL-Enhanced Player",
        agent_a_cls=RLEnhancedPlayer,
        agent_a_kwargs=rl_kwargs,
        agent_b_name="Random Baseline (P25CS0004)",
        agent_b_cls=RandomPlayer,
        agent_b_kwargs={},
        num_games=args.games_per_baseline,
        max_plies=args.max_plies
    )
    results["vs_random"] = res_rand

    # 2. vs Greedy Material Baseline
    res_greedy = run_benchmark_series(
        agent_a_name="RL-Enhanced Player",
        agent_a_cls=RLEnhancedPlayer,
        agent_a_kwargs=rl_kwargs,
        agent_b_name="Greedy Material Player",
        agent_b_cls=GreedyMaterialPlayer,
        agent_b_kwargs={},
        num_games=args.games_per_baseline,
        max_plies=args.max_plies
    )
    results["vs_greedy"] = res_greedy

    # 3. vs Untrained RL Agent (Ablation: Search with theta_0)
    res_untrained = run_benchmark_series(
        agent_a_name="RL-Enhanced Player",
        agent_a_cls=RLEnhancedPlayer,
        agent_a_kwargs=rl_kwargs,
        agent_b_name="Untrained RL Agent (theta_0)",
        agent_b_cls=UntrainedRLPlayer,
        agent_b_kwargs={"depth": 2},
        num_games=args.games_per_baseline,
        max_plies=args.max_plies
    )
    results["vs_untrained"] = res_untrained

    # 4. vs Heuristic Alpha-Beta Player
    res_heuristic = run_benchmark_series(
        agent_a_name="RL-Enhanced Player",
        agent_a_cls=RLEnhancedPlayer,
        agent_a_kwargs=rl_kwargs,
        agent_b_name="Heuristic Alpha-Beta Player",
        agent_b_cls=HeuristicPlayer,
        agent_b_kwargs={"depth": 2},
        num_games=args.games_per_baseline,
        max_plies=args.max_plies
    )
    results["vs_heuristic"] = res_heuristic

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Benchmark evaluation results successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
