"""
Chess Piece Module

Concept: A chess piece is an entity that moves according to specific rules
differentiated by type (Pawn, Rook, Knight, etc.) and color (White/Black).

In object-oriented programming, we represent pieces as objects with:
- A type that determines movement and capturing rules
- A color that determines which player owns the piece
"""

from enum import Enum, auto
from typing import NamedTuple, Literal, cast


# =============================================================================
# Type Aliases for Readability
# =============================================================================
# White means a piece belongs to the white player
# Black means a piece belongs to the black player
Color = Literal['w', 'b']  # Using 'w' and 'b' as standard notation


# =============================================================================
# Piece Types
# =============================================================================
# Chess has 6 piece types, each with unique movement and capture rules:
# 1. Pawn - moves forward 1 (or 2 on first move), captures diagonally
# 2. Rook - moves horizontally/vertically any distance
# 3. Knight - moves in L-shape (2+1 squares), can jump
# 4. Bishop - moves diagonally any distance
# 5. Queen - combines Rook and Bishop movement (most powerful)
# 6. King - moves 1 square in any direction (most critical piece)
class PieceType(str, Enum):
    PAWN = 'p'  # Pedestrian - slow, captures diagonally
    ROOK = 'r'  # Castle-like, controls files and ranks
    KNIGHT = 'n'  # Horse - jumps, only non-capturing piece that can
    BISHOP = 'b'  # Sliding diagonally, blocked by other pieces
    QUEEN = 'q'  # Queen - slides in all directions (most powerful)
    KING = 'k'   # Monarch - moves 1 square, cannot capture own pieces


# =============================================================================
# Representation of a chess piece
# =============================================================================
# Use NamedTuple for immutable records (efficient and type-safe)
# Each piece has a type and color that never change during the game
class Piece(NamedTuple):
    """
    Immutable chess piece with type and color.
    
    NamedTuple is used because:
    1. Pieces are immutable - they don't change type or color during a game
    2. NamedTuples are efficient (stored as tuples, not full objects)
    3. Clear and explicit about the data structure
    
    Note: If we needed mutable pieces (e.g., promoting a pawn to queen),
    we would need to use class-based objects with __setattr__.
    """
    type: PieceType  # The kind of piece (p, r, n, b, q, k)
    color: Color     # Which player owns it ('w' for white, 'b' for black)
    
    def is_white(self) -> bool:
        """Check if this piece belongs to white player."""
        return self.color == 'w'
    
    def is_black(self) -> bool:
        """Check if this piece belongs to black player."""
        return self.color == 'b'
    
    def __repr__(self) -> str:
        """Debug representation showing piece type and color."""
        return f'{self.color} {self.type.value.upper()}'
    
    def __str__(self) -> str:
        """Friendly representation of piece (e.g., 'P' for white pawn)."""
        symbols = {
            'p': 'P', 'r': 'R', 'n': 'N', 'b': 'B', 'q': 'Q', 'k': 'K'
        }
        return symbols[self.type.value]


# =============================================================================
# Board Square Coordinates
# =============================================================================
# Chess board uses 8x8 grid with ranks (1-8) and files (a-h)
# Standard notation: a1, h8, etc.
Square = tuple[int, int]  # (row, col) - row 0=rank 8, col 0=file a


# =============================================================================
# Board Representation
# =============================================================================
# A chess board is an 8x8 grid of squares
# None means empty square, Piece means occupied square
class Board:
    """
    8x8 chess board represented as 2D array.
    
    Board coordinates:
        a b c d  e  f  g  h
      8 [ ][ ][ ][ ][ ][ ][ ][ ]  # row 0 in array
        [ ][ ][ ][ ][ ][ ][ ][ ]  # row 1
        ...
        [ ][ ][ ][ ][ ][ ][ ][ ]  # row 7
        a b c d  e  f  g  h       # col 0 in array
    
    Array index to Chess notation:
        row = 7 - square_row (rank 8 is index 0, rank 1 is index 7)
        col = square_col       (file a is index 0, file h is index 7)
    """
    
    BOARD_SIZE = 8  # Chess board is always 8x8
    
    def __init__(self) -> None:
        """Initialize empty board with all squares containing None."""
        # Two-dimensional array of 8 rows and 8 columns
        # Board[row][col] where row 0 = rank 8, col 0 = file a
        self._squares: list[list[Piece | None]] = [
            [None] * Board.BOARD_SIZE for _ in range(Board.BOARD_SIZE)
        ]
    
    @property
    def squares(self) -> list[list[Piece | None]]:
        """
        Access the board as 2D array.
        
        board[row][col]:
            - Row 0: Rank 8 (white player's starting side)
            - Row 7: Rank 1 (black player's starting side)
            - Col 0: File a
            - Col 7: File h
        """
        return self._squares
    
    def get(self, square: Square) -> Piece | None:
        """
        Get the piece at a given square.
        
        Args:
            square: (row, col) coordinate on the board (0-7, 0-7)
        
        Returns:
            Piece object at square, or None if empty
        """
        row, col = square
        # Validate square is on board
        if not (0 <= row < Board.BOARD_SIZE and 0 <= col < Board.BOARD_SIZE):
            raise ValueError(f"Invalid square: {square}")
        return self._squares[row][col]
    
    def set(self, square: Square, piece: Piece | None) -> None:
        """
        Place or remove a piece from a square.
        
        Args:
            square: (row, col) coordinate
            piece: Piece to place, or None to remove
        
        Raises:
            ValueError: If piece already exists at target square
        """
        row, col = square
        # Validate square
        if not (0 <= row < Board.BOARD_SIZE and 0 <= col < Board.BOARD_SIZE):
            raise ValueError(f"Invalid square: {square}")
        
        # Check for conflicts (can't place two pieces on same square)
        if piece is not None and self.get(square) is not None:
            raise ValueError(f"Piece already exists at {square}: {self.get(square)}")
        
        self._squares[row][col] = piece
    
    def is_empty(self, square: Square) -> bool:
        """Check if a square contains no pieces."""
        return self.get(square) is None
    
    def to_coords(self, square: Square) -> Square:
        """
        Convert chess notation to array coordinates.
        
        Example:
            to_coords('a1') -> (7, 0)
            to_coords('e4') -> (3, 4)
        """
        # Files a-h map to columns 0-7
        file_to_col: dict[str, int] = {c: i for i, c in enumerate('abcdefgh')}
        
        # Rank is 1-indexed, array is 0-indexed (rank 8 is index 0)
        ranks: dict[str, int] = {'1': 7, '2': 6, '3': 5, '4': 4,
                                '5': 3, '6': 2, '7': 1, '8': 0}
        
        col = file_to_col[square[0]]
        row = ranks[square[1]]
        return row, col
    
    @staticmethod
    def notation_to_coords(square: Square) -> Square:
        """
        Convert algebraic notation square to array coordinates.
        
        Example:
            notation_to_coords('a1') -> (7, 0)
            notation_to_coords('e4') -> (3, 4)
        """
        # Files a-h map to columns 0-7
        file_to_col: dict[str, int] = {c: i for i, c in enumerate('abcdefgh')}
        
        # Rank is 1-indexed, array is 0-indexed (rank 8 is index 0)
        ranks: dict[str, int] = {'1': 7, '2': 6, '3': 5, '4': 4,
                                '5': 3, '6': 2, '7': 1, '8': 0}
        
        col = file_to_col[square[0]]
        row = ranks[square[1]]
        return row, col
    
    @classmethod
    def from_notation(cls) -> 'Board':
        """
        Create a board from algebraic notation.
        
        Standard starting position:
        White pieces:
            - Rooks on a1, h1
            - Knights on b1, g1
            - Bishops on c1, f1
            - Queen on d1
            - King on e1
            - Pawns on rank 2
        
        Black pieces:
            - Mirror of white on rank 7 and 8
        """
        board = cls()
        
        # Initial piece placement for white
        initial_white: dict[str, Piece] = {
            'a1': Piece('r', 'w'), 'b1': Piece('n', 'w'),
            'c1': Piece('b', 'w'), 'd1': Piece('q', 'w'),
            'e1': Piece('k', 'w'), 'f1': Piece('b', 'w'),
            'g1': Piece('n', 'w'), 'h1': Piece('r', 'w')
        }
        
        # Initial piece placement for black
        initial_black: dict[str, Piece] = {
            'a8': Piece('r', 'b'), 'b8': Piece('n', 'b'),
            'c8': Piece('b', 'b'), 'd8': Piece('q', 'b'),
            'e8': Piece('k', 'b'), 'f8': Piece('b', 'b'),
            'g8': Piece('n', 'b'), 'h8': Piece('r', 'b')
        }
        
        # Set up white pieces
        for square_str, piece in initial_white.items():
            file_str, rank = square_str[0], int(square_str[1])
            col = 'abcdefgh'.index(file_str)
            row = 8 - rank
            board.set((row, col), piece)
        
        # Set up black pieces
        for square_str, piece in initial_black.items():
            file_str, rank = square_str[0], int(square_str[1])
            col = 'abcdefgh'.index(file_str)
            row = 8 - rank
            board.set((row, col), piece)
        
        return board
