"""
Feature representation and state space extractor for 8x8 Chess RL Engine.
Defines the mapping phi(s) in R^D that represents chess positions as normalized,
strategically meaningful features for Reinforcement Learning (TD-Leaf / TD(lambda)).
"""

import chess
from config import (
    BOARD_WIDTH,
    BOARD_HEIGHT,
    EMPTY_SQUARE,
    WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
    BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING,
    PAWN_PST, KNIGHT_PST, BISHOP_PST, ROOK_PST, QUEEN_PST,
    KING_PST_MIDDLE_GAME, KING_PST_LATE_GAME,
)
from board import sq_to_rc, rc_to_sq

FEATURE_NAMES = [
    # Material counts
    "white_pawns", "white_knights", "white_bishops", "white_rooks", "white_queens",
    "black_pawns", "black_knights", "black_bishops", "black_rooks", "black_queens",
    "net_pawns", "net_knights", "net_bishops", "net_rooks", "net_queens",
    "game_phase_endgame",
    # Positional PST terms (White, Black, Net)
    "pst_pawn_w", "pst_knight_w", "pst_bishop_w", "pst_rook_w", "pst_queen_w", "pst_king_w",
    "pst_pawn_b", "pst_knight_b", "pst_bishop_b", "pst_rook_b", "pst_queen_b", "pst_king_b",
    "net_pst_pawns", "net_pst_knights", "net_pst_bishops", "net_pst_rooks", "net_pst_queens", "net_pst_kings",
    # Pawn structure
    "passed_pawns_w", "passed_pawns_b", "net_passed_pawns",
    "doubled_pawns_w", "doubled_pawns_b", "net_doubled_pawns",
    "isolated_pawns_w", "isolated_pawns_b", "net_isolated_pawns",
    "connected_pawns_w", "connected_pawns_b", "net_connected_pawns",
    "center_pawns_w", "center_pawns_b", "net_center_pawns",
    # Mobility & Central Control
    "mobility_w", "mobility_b", "net_mobility",
    "center_control_w", "center_control_b", "net_center_control",
    # King safety & Checks
    "in_check_w", "in_check_b",
    "king_pawn_shield_w", "king_pawn_shield_b", "net_king_shield",
    "king_tropism_w", "king_tropism_b",
    # Castling & Minor Piece Development
    "castling_rights_w", "castling_rights_b",
    "undeveloped_minors_w", "undeveloped_minors_b", "net_undeveloped_minors",
    "bishop_pair_w", "bishop_pair_b", "net_bishop_pair",
    # Constant Bias
    "bias"
]

FEATURE_DIM = len(FEATURE_NAMES)


class FeatureExtractor:
    """
    Extracts a dense feature vector phi(s) in R^FEATURE_DIM from a chess position.
    All features are computed from White's perspective (+ means White is advantageous).
    """

    def __init__(self):
        self.dim = FEATURE_DIM
        self.feature_names = FEATURE_NAMES

    def extract_features(self, engine):
        """
        Extract normalized feature vector from GameEngine or chess.Board.
        Returns a list of float features.
        """
        cb = engine.chess_board if hasattr(engine, "chess_board") else engine

        features = [0.0] * self.dim
        idx = 0

        # -------------------------------------------------------------
        # 1. Material counts & balance
        # -------------------------------------------------------------
        wp = len(cb.pieces(chess.PAWN, chess.WHITE))
        wn = len(cb.pieces(chess.KNIGHT, chess.WHITE))
        wb = len(cb.pieces(chess.BISHOP, chess.WHITE))
        wr = len(cb.pieces(chess.ROOK, chess.WHITE))
        wq = len(cb.pieces(chess.QUEEN, chess.WHITE))

        bp = len(cb.pieces(chess.PAWN, chess.BLACK))
        bn = len(cb.pieces(chess.KNIGHT, chess.BLACK))
        bb = len(cb.pieces(chess.BISHOP, chess.BLACK))
        br = len(cb.pieces(chess.ROOK, chess.BLACK))
        bq = len(cb.pieces(chess.QUEEN, chess.BLACK))

        features[idx] = wp / 8.0; idx += 1
        features[idx] = wn / 2.0; idx += 1
        features[idx] = wb / 2.0; idx += 1
        features[idx] = wr / 2.0; idx += 1
        features[idx] = wq / 1.0; idx += 1

        features[idx] = bp / 8.0; idx += 1
        features[idx] = bn / 2.0; idx += 1
        features[idx] = bb / 2.0; idx += 1
        features[idx] = br / 2.0; idx += 1
        features[idx] = bq / 1.0; idx += 1

        features[idx] = (wp - bp) / 8.0; idx += 1
        features[idx] = (wn - bn) / 2.0; idx += 1
        features[idx] = (wb - bb) / 2.0; idx += 1
        features[idx] = (wr - br) / 2.0; idx += 1
        features[idx] = (wq - bq) / 1.0; idx += 1

        # Non-pawn material (endgame phase factor)
        # Total starting non-pawn material: 2*3.2 + 2*3.3 + 2*5.0 + 9.0 = 32.0 per side (64 total)
        total_non_pawn = (wn + bn) * 3 + (wb + bb) * 3 + (wr + br) * 5 + (wq + bq) * 9
        endgame_factor = max(0.0, min(1.0, 1.0 - (total_non_pawn / 48.0)))
        features[idx] = endgame_factor; idx += 1

        # -------------------------------------------------------------
        # 2. Piece-Square Tables (PST)
        # -------------------------------------------------------------
        pst_w = [0.0] * 6
        pst_b = [0.0] * 6

        # Pawns
        for sq in cb.pieces(chess.PAWN, chess.WHITE):
            r, c = sq_to_rc(sq)
            pst_w[0] += PAWN_PST[r][c]
        for sq in cb.pieces(chess.PAWN, chess.BLACK):
            r, c = sq_to_rc(sq)
            pst_b[0] += PAWN_PST[7 - r][c]

        # Knights
        for sq in cb.pieces(chess.KNIGHT, chess.WHITE):
            r, c = sq_to_rc(sq)
            pst_w[1] += KNIGHT_PST[r][c]
        for sq in cb.pieces(chess.KNIGHT, chess.BLACK):
            r, c = sq_to_rc(sq)
            pst_b[1] += KNIGHT_PST[7 - r][c]

        # Bishops
        for sq in cb.pieces(chess.BISHOP, chess.WHITE):
            r, c = sq_to_rc(sq)
            pst_w[2] += BISHOP_PST[r][c]
        for sq in cb.pieces(chess.BISHOP, chess.BLACK):
            r, c = sq_to_rc(sq)
            pst_b[2] += BISHOP_PST[7 - r][c]

        # Rooks
        for sq in cb.pieces(chess.ROOK, chess.WHITE):
            r, c = sq_to_rc(sq)
            pst_w[3] += ROOK_PST[r][c]
        for sq in cb.pieces(chess.ROOK, chess.BLACK):
            r, c = sq_to_rc(sq)
            pst_b[3] += ROOK_PST[7 - r][c]

        # Queens
        for sq in cb.pieces(chess.QUEEN, chess.WHITE):
            r, c = sq_to_rc(sq)
            pst_w[4] += QUEEN_PST[r][c]
        for sq in cb.pieces(chess.QUEEN, chess.BLACK):
            r, c = sq_to_rc(sq)
            pst_b[4] += QUEEN_PST[7 - r][c]

        # Kings (interpolating between middle and late game)
        w_king_sq = cb.king(chess.WHITE)
        if w_king_sq is not None:
            r, c = sq_to_rc(w_king_sq)
            mg_val = KING_PST_MIDDLE_GAME[r][c]
            eg_val = KING_PST_LATE_GAME[r][c]
            pst_w[5] += (1.0 - endgame_factor) * mg_val + endgame_factor * eg_val

        b_king_sq = cb.king(chess.BLACK)
        if b_king_sq is not None:
            r, c = sq_to_rc(b_king_sq)
            mg_val = KING_PST_MIDDLE_GAME[7 - r][c]
            eg_val = KING_PST_LATE_GAME[7 - r][c]
            pst_b[5] += (1.0 - endgame_factor) * mg_val + endgame_factor * eg_val

        # Scale PST values to reasonable numerical ranges (~[-1, 1])
        scale_factors = [100.0, 80.0, 80.0, 60.0, 60.0, 60.0]
        for i in range(6):
            features[idx] = pst_w[i] / scale_factors[i]; idx += 1
        for i in range(6):
            features[idx] = pst_b[i] / scale_factors[i]; idx += 1
        for i in range(6):
            features[idx] = (pst_w[i] - pst_b[i]) / scale_factors[i]; idx += 1

        # -------------------------------------------------------------
        # 3. Pawn Structure Analysis
        # -------------------------------------------------------------
        w_pawn_files = [0] * 8
        b_pawn_files = [0] * 8
        w_pawn_sqs = list(cb.pieces(chess.PAWN, chess.WHITE))
        b_pawn_sqs = list(cb.pieces(chess.PAWN, chess.BLACK))

        for sq in w_pawn_sqs:
            w_pawn_files[chess.square_file(sq)] += 1
        for sq in b_pawn_sqs:
            b_pawn_files[chess.square_file(sq)] += 1

        # Passed pawns
        passed_w = 0
        for sq in w_pawn_sqs:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            opp_blockers = [
                s for s in b_pawn_sqs
                if abs(chess.square_file(s) - f) <= 1 and chess.square_rank(s) > r
            ]
            if not opp_blockers:
                passed_w += 1

        passed_b = 0
        for sq in b_pawn_sqs:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            opp_blockers = [
                s for s in w_pawn_sqs
                if abs(chess.square_file(s) - f) <= 1 and chess.square_rank(s) < r
            ]
            if not opp_blockers:
                passed_b += 1

        features[idx] = passed_w / 4.0; idx += 1
        features[idx] = passed_b / 4.0; idx += 1
        features[idx] = (passed_w - passed_b) / 4.0; idx += 1

        # Doubled pawns
        doubled_w = sum(max(0, count - 1) for count in w_pawn_files)
        doubled_b = sum(max(0, count - 1) for count in b_pawn_files)
        features[idx] = doubled_w / 4.0; idx += 1
        features[idx] = doubled_b / 4.0; idx += 1
        features[idx] = (doubled_b - doubled_w) / 4.0; idx += 1  # Less doubled pawns is better for White

        # Isolated pawns
        isolated_w = 0
        isolated_b = 0
        for f in range(8):
            if w_pawn_files[f] > 0:
                has_left = (f > 0 and w_pawn_files[f - 1] > 0)
                has_right = (f < 7 and w_pawn_files[f + 1] > 0)
                if not has_left and not has_right:
                    isolated_w += w_pawn_files[f]
            if b_pawn_files[f] > 0:
                has_left = (f > 0 and b_pawn_files[f - 1] > 0)
                has_right = (f < 7 and b_pawn_files[f + 1] > 0)
                if not has_left and not has_right:
                    isolated_b += b_pawn_files[f]

        features[idx] = isolated_w / 4.0; idx += 1
        features[idx] = isolated_b / 4.0; idx += 1
        features[idx] = (isolated_b - isolated_w) / 4.0; idx += 1

        # Connected pawns
        connected_w = 0
        for sq in w_pawn_sqs:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            if any(chess.square_file(s) in (f - 1, f + 1) and abs(chess.square_rank(s) - r) <= 1 for s in w_pawn_sqs):
                connected_w += 1
        connected_b = 0
        for sq in b_pawn_sqs:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            if any(chess.square_file(s) in (f - 1, f + 1) and abs(chess.square_rank(s) - r) <= 1 for s in b_pawn_sqs):
                connected_b += 1

        features[idx] = connected_w / 8.0; idx += 1
        features[idx] = connected_b / 8.0; idx += 1
        features[idx] = (connected_w - connected_b) / 8.0; idx += 1

        # Center pawns (d4/d5/e4/e5)
        center_sqs_w = {chess.D4, chess.E4, chess.D3, chess.E3}
        center_sqs_b = {chess.D5, chess.E5, chess.D6, chess.E6}
        cp_w = sum(1 for sq in w_pawn_sqs if sq in center_sqs_w)
        cp_b = sum(1 for sq in b_pawn_sqs if sq in center_sqs_b)
        features[idx] = cp_w / 4.0; idx += 1
        features[idx] = cp_b / 4.0; idx += 1
        features[idx] = (cp_w - cp_b) / 4.0; idx += 1

        # -------------------------------------------------------------
        # 4. Mobility & Central Control
        # -------------------------------------------------------------
        # Pseudo-legal or legal moves approximation for side to move
        # To avoid expensive full legal move generation from both colors simultaneously:
        white_mobility = 0
        black_mobility = 0
        
        # Center square attacks {d4, d5, e4, e5, c4, c5, f4, f5}
        center_target_squares = [chess.D4, chess.D5, chess.E4, chess.E5, chess.C4, chess.C5, chess.F4, chess.F5]
        center_ctrl_w = 0
        center_ctrl_b = 0
        for sq in center_target_squares:
            center_ctrl_w += len(cb.attackers(chess.WHITE, sq))
            center_ctrl_b += len(cb.attackers(chess.BLACK, sq))

        # Rapid mobility estimation from attack bitboards
        for sq in cb.pieces(chess.KNIGHT, chess.WHITE):
            white_mobility += len(cb.attacks(sq))
        for sq in cb.pieces(chess.BISHOP, chess.WHITE):
            white_mobility += len(cb.attacks(sq))
        for sq in cb.pieces(chess.ROOK, chess.WHITE):
            white_mobility += len(cb.attacks(sq))
        for sq in cb.pieces(chess.QUEEN, chess.WHITE):
            white_mobility += len(cb.attacks(sq))

        for sq in cb.pieces(chess.KNIGHT, chess.BLACK):
            black_mobility += len(cb.attacks(sq))
        for sq in cb.pieces(chess.BISHOP, chess.BLACK):
            black_mobility += len(cb.attacks(sq))
        for sq in cb.pieces(chess.ROOK, chess.BLACK):
            black_mobility += len(cb.attacks(sq))
        for sq in cb.pieces(chess.QUEEN, chess.BLACK):
            black_mobility += len(cb.attacks(sq))

        features[idx] = white_mobility / 50.0; idx += 1
        features[idx] = black_mobility / 50.0; idx += 1
        features[idx] = (white_mobility - black_mobility) / 50.0; idx += 1

        features[idx] = center_ctrl_w / 16.0; idx += 1
        features[idx] = center_ctrl_b / 16.0; idx += 1
        features[idx] = (center_ctrl_w - center_ctrl_b) / 16.0; idx += 1

        # -------------------------------------------------------------
        # 5. King Safety & Checks
        # -------------------------------------------------------------
        in_check_w = 1.0 if (cb.turn == chess.WHITE and cb.is_check()) else 0.0
        in_check_b = 1.0 if (cb.turn == chess.BLACK and cb.is_check()) else 0.0
        features[idx] = in_check_w; idx += 1
        features[idx] = in_check_b; idx += 1

        # King pawn shield: count pawns in front of castled king
        shield_w = 0
        if w_king_sq is not None:
            kf = chess.square_file(w_king_sq)
            kr = chess.square_rank(w_king_sq)
            if kr <= 2:
                shield_w = sum(1 for sq in w_pawn_sqs if abs(chess.square_file(sq) - kf) <= 1 and 0 < chess.square_rank(sq) - kr <= 2)
        
        shield_b = 0
        if b_king_sq is not None:
            kf = chess.square_file(b_king_sq)
            kr = chess.square_rank(b_king_sq)
            if kr >= 5:
                shield_b = sum(1 for sq in b_pawn_sqs if abs(chess.square_file(sq) - kf) <= 1 and 0 < kr - chess.square_rank(sq) <= 2)

        features[idx] = shield_w / 3.0; idx += 1
        features[idx] = shield_b / 3.0; idx += 1
        features[idx] = (shield_w - shield_b) / 3.0; idx += 1

        # King tropism: distance of enemy major pieces to friendly king
        tropism_w = 0.0
        if w_king_sq is not None:
            kr, kf = chess.square_rank(w_king_sq), chess.square_file(w_king_sq)
            for sq in list(cb.pieces(chess.QUEEN, chess.BLACK)) + list(cb.pieces(chess.ROOK, chess.BLACK)):
                dist = abs(chess.square_rank(sq) - kr) + abs(chess.square_file(sq) - kf)
                tropism_w += (14 - dist)
        tropism_b = 0.0
        if b_king_sq is not None:
            kr, kf = chess.square_rank(b_king_sq), chess.square_file(b_king_sq)
            for sq in list(cb.pieces(chess.QUEEN, chess.WHITE)) + list(cb.pieces(chess.ROOK, chess.WHITE)):
                dist = abs(chess.square_rank(sq) - kr) + abs(chess.square_file(sq) - kf)
                tropism_b += (14 - dist)

        features[idx] = tropism_w / 28.0; idx += 1
        features[idx] = tropism_b / 28.0; idx += 1

        # -------------------------------------------------------------
        # 6. Castling Rights & Minor Piece Development
        # -------------------------------------------------------------
        castle_w = (1.0 if cb.has_kingside_castling_rights(chess.WHITE) else 0.0) + \
                   (1.0 if cb.has_queenside_castling_rights(chess.WHITE) else 0.0)
        castle_b = (1.0 if cb.has_kingside_castling_rights(chess.BLACK) else 0.0) + \
                   (1.0 if cb.has_queenside_castling_rights(chess.BLACK) else 0.0)
        features[idx] = castle_w / 2.0; idx += 1
        features[idx] = castle_b / 2.0; idx += 1

        # Undeveloped minors on rank 1 (White) and rank 8 (Black)
        undev_w = 0
        for sq in (chess.B1, chess.C1, chess.F1, chess.G1):
            p = cb.piece_at(sq)
            if p and p.color == chess.WHITE and p.piece_type in (chess.KNIGHT, chess.BISHOP):
                undev_w += 1
        undev_b = 0
        for sq in (chess.B8, chess.C8, chess.F8, chess.G8):
            p = cb.piece_at(sq)
            if p and p.color == chess.BLACK and p.piece_type in (chess.KNIGHT, chess.BISHOP):
                undev_b += 1

        features[idx] = undev_w / 4.0; idx += 1
        features[idx] = undev_b / 4.0; idx += 1
        features[idx] = (undev_b - undev_w) / 4.0; idx += 1  # Less undeveloped is better for White

        bp_w = 1.0 if wb >= 2 else 0.0
        bp_b = 1.0 if bb >= 2 else 0.0
        features[idx] = bp_w; idx += 1
        features[idx] = bp_b; idx += 1
        features[idx] = (bp_w - bp_b); idx += 1

        # -------------------------------------------------------------
        # 7. Constant Bias
        # -------------------------------------------------------------
        features[idx] = 1.0; idx += 1

        return features
