"""
Chess Attack Detection

This module contains functions for checking if squares are under attack.
Used for check detection and validating castling moves.

Concept: To check if a square is under attack by a color, we check if any
piece of that color can "capture" on that square.
"""

from .piece import Board, Piece, PieceType, Color
from .move import rook_moves, knight_moves, bishop_moves, queen_moves, king_moves


# =============================================================================
# Attack Detection Functions
# =============================================================================
def is_square_attacked(square: tuple[int, int], attacker_color: Color, 
                      board: Board) -> bool:
    """
    Check if a square is under attack by the specified player.
    
    This checks all piece types to see if they can attack this square.
    
    Parameters:
        square: (row, col) coordinate to check
        attacker_color: The color that would be attacking ('w' or 'b')
        board: Current board state
    
    Returns:
        True if the square is under attack, False otherwise
    
    Example:
        # Check if d5 is under attack by white's queen
        is_square_attacked((3, 3), 'w', board)
    """
    # We need to check if any opponent piece can capture on this square
    # This means: can any piece of 'attacker_color' reach 'square'?
    
    for row in range(Board.BOARD_SIZE):
        for col in range(Board.BOARD_SIZE):
            piece = board.get((row, col))
            
            # Skip if no piece here, or piece doesn't match attacker's color
            if piece is None or piece.color != attacker_color:
                continue
            
            # Get movement directions for this piece type
            if piece.type == PieceType.PAWN:
                moves = pawn_attack_moves(board, piece)
            elif piece.type == PieceType.ROOK:
                moves = rook_moves(board, piece)
            elif piece.type == PieceType.KNIGHT:
                moves = knight_moves(board, piece)
            elif piece.type == PieceType.BISHOP:
                moves = bishop_moves(board, piece)
            elif piece.type == PieceType.QUEEN:
                moves = queen_moves(board, piece)
            elif piece.type == PieceType.KING:
                moves = king_moves(board, piece)
            else:
                continue  # Unknown piece type
            
            # Check if any move direction reaches the target square
            for dr, dc in moves:
                target_row, target_col = row + dr, col + dc
                target_sq = (target_row, target_col)
                
                # Check bounds
                if not (0 <= target_row < Board.BOARD_SIZE and 
                        0 <= target_col < Board.BOARD_SIZE):
                    continue
                
                # Check if this is an attack (we're not blocked by own pieces)
                # For simplicity, we just check if the direction reaches
                if target_sq == square:
                    return True
    
    return False


# =============================================================================
# Pawn Attack Direction
# =============================================================================
def pawn_attack_moves(board: Board, piece: Piece) -> list[tuple[int, int]]:
    """
    Get the direction a pawn attacks (captures diagonally).
    
    Pawn rules:
    - White pawns attack diagonally up (row - 1)
    - Black pawns attack diagonally down (row + 1)
    - Both attack left and right diagonally
    
    Returns:
        List of (row_delta, col_delta) direction tuples
    
    Example:
        White pawn attacks: [(-1, -1), (-1, 1)]  # Up-left, Up-right
        Black pawn attacks: [(1, -1), (1, 1)]   # Down-left, Down-right
    """
    if piece.color == 'w':
        # White pawn attacks UP diagonally
        return [(-1, -1), (-1, 1)]
    else:
        # Black pawn attacks DOWN diagonally
        return [(1, -1), (1, 1)]


# =============================================================================
# King Detection
# =============================================================================
def find_king(board: Board, player_color: Color) -> tuple[int, int] | None:
    """
    Find the position of a player's king on the board.
    
    Parameters:
        board: Current board state
        player_color: 'w' for white, 'b' for black
    
    Returns:
        (row, col) of the king, or None if not found
    
    Example:
        # Find white king
        white_king_pos = find_king(board, 'w')
        print(f"White king is at: {white_king_pos}")  # e.g., (0, 4) for e1
    """
    for row in range(Board.BOARD_SIZE):
        for col in range(Board.BOARD_SIZE):
            piece = board.get((row, col))
            if piece is not None and piece.type == PieceType.KING and piece.color == player_color:
                return (row, col)
    return None  # King not found (shouldn't happen in legal games)


# =============================================================================
# Check Detection
# =============================================================================
def is_in_check(board: Board, player: Color) -> bool:
    """
    Check if the specified player's king is in check.
    
    A player is in check if their king is under attack by the opponent.
    
    Parameters:
        board: Current board state
        player: The player to check ('w' or 'b')
    
    Returns:
        True if the player is in check, False otherwise
    
    Example:
        # Is white in check?
        if is_in_check(board, 'w'):
            print("Check!")
    """
    king_pos = find_king(board, player)
    
    if king_pos is None:
        return False  # King was captured (shouldn't happen in legal games)
    
    # Check if opponent can attack our king's square
    return is_square_attacked(king_pos, player.opposite(), board)


class PlayerColor:
    """
    Wrapper class for player colors with helper methods.
    """
    WHITE = 'w'
    BLACK = 'b'
    
    @classmethod
    def opposite(cls, color: str) -> str:
        """Get the opposite player's color."""
        return 'b' if color == 'w' else 'w'
