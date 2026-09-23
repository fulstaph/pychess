"""
Chess Engine in Python with Strict Typing

A simple, type-safe chess engine implementation in Python.
Designed for clarity and educational purposes.

Usage:
    from chess import ChessGame, Board, Piece
    
    # Create a new game
    game = ChessGame()
    
    # Make a move
    move = "e2e4"
    game.make_move(move)
    
    # Check game status
    print(game.is_check())
    print(game.is_checkmate())
    
    # View the board
    chess.print_board(game.get_board())

Modules:
    piece.py - Piece definitions and board representation
    move.py - Move generation, validation, and notation
    game.py - Game engine with check, checkmate, stalemate detection

Chess Concepts:
    1. Piece Types - 6 types with unique movement rules
    2. Board Representation - 8x8 grid using 2D array
    3. Movement - Each piece type has specific valid moves
    4. Castling - Special move for King (O-O, O-O-O)
    5. En Passant - Special pawn capture
    6. Promotion - Pawn reaches end of board
    7. Check - King under attack
    8. Checkmate - King in check with no legal moves
    9. Stalemate - No legal moves, but not in check
"""

from .piece import Piece, PieceType, Board, Square
from .move import Move, GameResult, file_letter, rank_letter
from .game import ChessGame, GameStatus, print_board
from .helpers import PlayerColor, notation_to_coords, square_notation

__all__ = [
    'Piece', 'PieceType', 'Color', 'Board', 'Square',
    'Move', 'GameResult', 'file_letter', 'rank_letter',
    'ChessGame', 'GameStatus', 'PlayerColor', 'print_board',
    'notation_to_coords', 'square_notation'
]

__version__ = '0.1.0'
__author__ = 'Chess Engine Dev'