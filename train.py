"""
Training script for Reinforcement Learning-Enhanced Chess Engine.
Runs TD-Leaf(lambda) self-play training loop, logs metrics, and saves weights.
"""

import argparse
import json
import os
import time

from config import (
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA,
    DEFAULT_ALPHA,
    DEFAULT_EPSILON,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_MIN_EPSILON,
    DEFAULT_EPISODES,
)
from rl_value_net import LinearValueFunction
from td_learning import TDTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Chess RL Value Function via TD-Leaf(lambda) Self-Play")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Number of self-play training episodes")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Learning rate alpha")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Discount factor gamma")
    parser.add_argument("--lam", type=float, default=DEFAULT_LAMBDA, help="Eligibility trace decay lambda")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="Initial exploration rate epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=DEFAULT_EPSILON_DECAY, help="Exploration decay factor")
    parser.add_argument("--min-epsilon", type=float, default=DEFAULT_MIN_EPSILON, help="Minimum exploration rate")
    parser.add_argument("--max-plies", type=int, default=120, help="Maximum plies per self-play game")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging frequency in episodes")
    parser.add_argument("--output-weights", type=str, default="trained_weights.json", help="Path to save learned weights")
    parser.add_argument("--output-metrics", type=str, default="training_metrics.json", help="Path to save training metrics")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoint weights")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("  REINFORCEMENT LEARNING CHESS ENGINE - TD-LEAF SELF-PLAY TRAINING  ")
    print("=" * 65)
    print(f"Episodes:           {args.episodes}")
    print(f"Learning Rate (a):  {args.alpha}")
    print(f"Discount (g):       {args.gamma}")
    print(f"Trace Decay (l):    {args.lam}")
    print(f"Exploration (eps):  {args.epsilon} -> min {args.min_epsilon} (decay {args.epsilon_decay})")
    print(f"Max Plies / Game:   {args.max_plies}")
    print(f"Output Weights:     {args.output_weights}")
    print("=" * 65)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize value function
    val_fn = LinearValueFunction(init_strategy="heuristic_prior")
    
    # Save initial untrained/prior weights as checkpoint 0
    init_ckpt = os.path.join(args.checkpoint_dir, "weights_ep0.json")
    val_fn.save_weights(init_ckpt)

    trainer = TDTrainer(
        value_fn=val_fn,
        gamma=args.gamma,
        lam=args.lam,
        alpha=args.alpha,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.min_epsilon,
    )

    # Multi-episode training with periodic checkpoints
    snapshot_interval = max(10, args.episodes // 5)
    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        loss, outcome, plies = trainer.train_episode(max_plies=args.max_plies)

        trainer.history["episodes"].append(ep)
        trainer.history["td_losses"].append(loss)
        trainer.history["outcomes"].append(outcome)
        trainer.history["game_lengths"].append(plies)
        trainer.history["epsilons"].append(trainer.epsilon)

        if outcome > 0.5:
            trainer.history["white_wins"] += 1
        elif outcome < -0.5:
            trainer.history["black_wins"] += 1
        else:
            trainer.history["draws"] += 1

        if ep % snapshot_interval == 0 or ep == args.episodes:
            ckpt_path = os.path.join(args.checkpoint_dir, f"weights_ep{ep}.json")
            val_fn.save_weights(ckpt_path)
            trainer.history["weight_history"].append({
                "episode": ep,
                "weights": val_fn.weights.copy().tolist(),
                "summary": val_fn.get_weights_summary()
            })

        if ep % args.log_interval == 0 or ep == 1:
            recent_losses = trainer.history["td_losses"][-args.log_interval:]
            recent_plies = trainer.history["game_lengths"][-args.log_interval:]
            avg_l = sum(recent_losses) / len(recent_losses)
            avg_p = sum(recent_plies) / len(recent_plies)
            w_win_pct = (trainer.history["white_wins"] / ep) * 100.0
            b_win_pct = (trainer.history["black_wins"] / ep) * 100.0
            draw_pct = (trainer.history["draws"] / ep) * 100.0
            elapsed = time.time() - start_time

            print(
                f"Ep {ep:4d}/{args.episodes} | "
                f"TD-Loss: {avg_l:7.4f} | "
                f"Avg Plies: {avg_p:5.1f} | "
                f"W/B/D: {w_win_pct:4.1f}%/{b_win_pct:4.1f}%/{draw_pct:4.1f}% | "
                f"Eps: {trainer.epsilon:.3f} | "
                f"Time: {elapsed:5.1f}s"
            )

    # Save final learned weights
    val_fn.save_weights(args.output_weights)
    print(f"\n[+] Successfully saved final learned weights to: {args.output_weights}")

    # Save training metrics JSON
    with open(args.output_metrics, "w") as f:
        json.dump(trainer.history, f, indent=2)
    print(f"[+] Successfully saved training metrics to: {args.output_metrics}")

    print("\nTraining complete! Key learned feature weights:")
    summary = val_fn.get_weights_summary()
    top_keys = ["net_pawns", "net_knights", "net_bishops", "net_rooks", "net_queens",
                "net_passed_pawns", "net_center_control", "net_mobility", "net_king_shield"]
    for k in top_keys:
        if k in summary:
            print(f"  - {k:22s}: {summary[k]:+.4f}")


if __name__ == "__main__":
    main()
