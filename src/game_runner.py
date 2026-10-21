"""
Game Runner for 8x8 Blitz Chess Matches.
Supports interactive and automated matches between RL agents and baseline players
with strict blitz clock enforcement (60s total time), board visualization, and summary logs.
"""

import argparse
import time
import sys

from board import GameEngine, Move
from config import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    EMPTY_SQUARE,
    BASE_PIECE_VALUES,
    UNICODE_PIECES,
    DEFAULT_BLITZ_TIME,
)
from rl_player import RLEnhancedPlayer
from baselines import RandomPlayer, GreedyMaterialPlayer, UntrainedRLPlayer, HeuristicPlayer


class PlayerClock:
    """Manages blitz countdown clocks for both players."""

    def __init__(self, total_seconds=DEFAULT_BLITZ_TIME):
        self.white_time = float(total_seconds)
        self.black_time = float(total_seconds)

    def format_time(self, seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        tenths = int((seconds - int(seconds)) * 10)
        return f"{mins:02d}:{secs:02d}.{tenths}"


def display_board(engine, clock=None, white_name="White", black_name="Black"):
    """Render the 8x8 chess board with Unicode pieces and side clock display."""
    file_labels = "    " + "   ".join(["a", "b", "c", "d", "e", "f", "g", "h"])
    horizontal_border = "  +" + "---+" * 8

    print("\n" + file_labels)
    print(horizontal_border)

    for r in range(BOARD_HEIGHT):
        rank = 8 - r
        row_str = f"{rank} |"
        for c in range(BOARD_WIDTH):
            piece = engine.board[r][c]
            sym = UNICODE_PIECES.get(piece, " ")
            row_str += f" {sym} |"
        row_str += f" {rank}"

        if clock is not None:
            if r == 1:
                row_str += f"    Black ({black_name}): {clock.format_time(clock.black_time)}"
            elif r == 6:
                row_str += f"    White ({white_name}): {clock.format_time(clock.white_time)}"

        print(row_str)
        print(horizontal_border)

    print(file_labels + "\n")


def run_blitz_game(white_cls, black_cls, white_kwargs=None, black_kwargs=None,
                   total_time_seconds=DEFAULT_BLITZ_TIME, quiet=False, max_turns=150):
    """
    Execute a full 8x8 Blitz chess game between two player classes.
    """
    white_kwargs = white_kwargs or {}
    black_kwargs = black_kwargs or {}

    engine = GameEngine()
    white_player = white_cls(engine, **white_kwargs)
    black_player = black_cls(engine, **black_kwargs)

    clock = PlayerClock(total_time_seconds)
    turn_counter = 0

    w_name = white_player.__class__.__name__
    b_name = black_player.__class__.__name__

    if not quiet:
        print("=" * 65)
        print(f"Starting Blitz Game (60s clock): {w_name} (White) vs {b_name} (Black)")
        print("=" * 65)
        display_board(engine, clock, w_name, b_name)

    game_over = False
    termination_reason = "ongoing"

    while not game_over and turn_counter < max_turns:
        state = engine.get_game_state()
        if state != "ongoing":
            game_over = True
            termination_reason = state
            break

        is_white = engine.white_to_move
        player = white_player if is_white else black_player
        color_tag = "White" if is_white else "Black"

        start_think = time.perf_counter()
        move = player.get_best_move()
        think_duration = time.perf_counter() - start_think

        # Decrement clock
        if is_white:
            clock.white_time -= think_duration
            if clock.white_time <= 0:
                termination_reason = "black_win_on_time"
                game_over = True
                break
        else:
            clock.black_time -= think_duration
            if clock.black_time <= 0:
                termination_reason = "white_win_on_time"
                game_over = True
                break

        if move is None:
            # No legal moves available
            termination_reason = "stalemate" if not engine.is_in_check() else "checkmate"
            game_over = True
            break

        engine.make_move(move)

        if not quiet:
            files = "abcdefgh"
            src = f"{files[move.start_col]}{8 - move.start_row}"
            dst = f"{files[move.end_col]}{8 - move.end_row}"
            promo = f"={move.promotion}" if move.promotion else ""
            cap = f" x {move.piece_captured}" if move.piece_captured != EMPTY_SQUARE else ""
            chk = " (Check!)" if engine.is_in_check() else ""
            nodes = getattr(player, "nodes_expanded", 1)
            print(f"Turn {turn_counter + 1:3d}: {color_tag:5s} plays {move.piece_moved} {src}->{dst}{promo}{cap}{chk} "
                  f"| Time: {think_duration:.3f}s | Nodes: {nodes}")
            if turn_counter % 5 == 0 or engine.is_in_check():
                display_board(engine, clock, w_name, b_name)

        turn_counter += 1

    final_state = engine.get_game_state()
    winner = None

    if termination_reason == "white_win_on_time":
        winner = "White"
        print(f"\n[!] Game Over: White wins on time! (Black flagged)")
    elif termination_reason == "black_win_on_time":
        winner = "Black"
        print(f"\n[!] Game Over: Black wins on time! (White flagged)")
    elif final_state == "checkmate":
        winner = "Black" if engine.white_to_move else "White"
        print(f"\n[!] Game Over: Checkmate! {winner} wins in {turn_counter} turns.")
    elif final_state in ("stalemate", "draw") or turn_counter >= max_turns:
        print(f"\n[!] Game Over: Draw ({final_state}) in {turn_counter} turns.")

    if not quiet:
        display_board(engine, clock, w_name, b_name)

    return {
        "winner": winner,
        "turns": turn_counter,
        "final_state": final_state,
        "white_time_left": clock.white_time,
        "black_time_left": clock.black_time,
    }


def parse_player(name):
    players = {
        "rl": RLEnhancedPlayer,
        "random": RandomPlayer,
        "greedy": GreedyMaterialPlayer,
        "untrained": UntrainedRLPlayer,
        "heuristic": HeuristicPlayer,
    }
    low = name.lower()
    if low in players:
        return players[low]
    raise ValueError(f"Unknown player type: {name}. Options: {list(players.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Run 8x8 Blitz Chess Match")
    parser.add_argument("--white", type=str, default="rl", help="White player: rl, random, greedy, untrained, heuristic")
    parser.add_argument("--black", type=str, default="heuristic", help="Black player: rl, random, greedy, untrained, heuristic")
    parser.add_argument("--time", type=float, default=DEFAULT_BLITZ_TIME, help="Total blitz clock time per player (seconds)")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode: suppress per-turn board printing")
    parser.add_argument("--turns", type=int, default=150, help="Maximum turns before draw")
    args = parser.parse_args()

    white_cls = parse_player(args.white)
    black_cls = parse_player(args.black)

    run_blitz_game(
        white_cls=white_cls,
        black_cls=black_cls,
        total_time_seconds=args.time,
        quiet=args.quiet,
        max_turns=args.turns
    )


if __name__ == "__main__":
    main()
