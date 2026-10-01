"""Standalone zero-mock UCI chess engine script for testing."""

import sys

from chess.engine import choose_move
from chess.game import ChessGame
from chess.helpers import notation_to_coords
from chess.piece import Board, Color, Piece, PieceType, Square
from chess.uci import to_uci


def parse_fen(fen_str: str) -> ChessGame:
    """Parse a basic FEN string into a ChessGame for test simulation."""
    tokens = fen_str.strip().split()
    board_part = tokens[0]
    turn: Color = "w" if len(tokens) > 1 and tokens[1] == "w" else "b"
    castling_token = tokens[2] if len(tokens) > 2 else "-"
    castling = frozenset(c for c in castling_token if c in "KQkq")
    ep_field = tokens[3] if len(tokens) > 3 else "-"
    en_passant: Square | None = None
    if ep_field != "-":
        try:
            en_passant = notation_to_coords(ep_field)
        except ValueError:
            en_passant = None

    ranks = board_part.split("/")
    changes: dict[Square, Piece | None] = {}
    for r, rank_str in enumerate(ranks):
        c = 0
        for ch in rank_str:
            if ch.isdigit():
                c += int(ch)
            else:
                pt = PieceType(ch.lower())
                color: Color = "w" if ch.isupper() else "b"
                changes[(r, c)] = Piece(pt, color)
                c += 1

    board = Board()._updated(changes)
    return ChessGame(
        board=board,
        turn=turn,
        castling_rights=castling,
        en_passant=en_passant,
    )


def main() -> None:
    game = ChessGame()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        if line == "uci":
            sys.stdout.write("id name FakeUCIEngine\n")
            sys.stdout.write("id author PychessTests\n")
            sys.stdout.write("option name Threads type spin default 1 min 1 max 128\n")
            sys.stdout.write(
                "option name Hash type spin default 16 min 1 max 1048576\n"
            )
            sys.stdout.write(
                "option name Skill Level type spin default 20 min 0 max 20\n"
            )
            sys.stdout.write("uciok\n")
            sys.stdout.flush()
        elif line == "isready":
            sys.stdout.write("readyok\n")
            sys.stdout.flush()
        elif line == "ucinewgame":
            game = ChessGame()
        elif line.startswith("setoption"):
            # Acknowledge option setting
            pass
        elif line.startswith("position"):
            tokens = line.split()
            moves_idx = tokens.index("moves") if "moves" in tokens else -1
            if "startpos" in tokens:
                game = ChessGame()
            elif "fen" in tokens:
                fen_tokens = tokens[2:moves_idx] if moves_idx != -1 else tokens[2:]
                game = parse_fen(" ".join(fen_tokens))

            if moves_idx != -1:
                for move_str in tokens[moves_idx + 1 :]:
                    game.make_move(move_str)
        elif line.startswith("go"):
            legal = game.legal_moves()
            if not legal:
                sys.stdout.write("bestmove 0000\n")
            else:
                try:
                    selected = choose_move(game, depth=1)
                except ValueError:
                    selected = legal[0]
                sys.stdout.write(f"bestmove {to_uci(selected)}\n")
            sys.stdout.flush()
        elif line == "quit":
            break


if __name__ == "__main__":
    main()
