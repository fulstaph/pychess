"""
Helper functions and classes for coordinate conversion.
"""

from typing import Literal
from enum import Enum


class PlayerColor(Enum):
    """Player colors."""
    WHITE = 'w'
    BLACK = 'b'
    
    @property
    def value(self) -> Literal['w', 'b']:
        return self.value
    
    def is_white(self) -> bool:
        return self == PlayerColor.WHITE
    
    def is_black(self) -> bool:
        return self == PlayerColor.BLACK
    
    def opposite(self) -> 'PlayerColor':
        return PlayerColor.BLACK if self == PlayerColor.WHITE else PlayerColor.WHITE


def notation_to_coords(square: str) -> tuple[int, int]:
    """
    Convert algebraic notation square to array coordinates.
    
    Args:
        square: Algebraic notation (e.g., 'e4', 'h8')
    
    Returns:
        (row, col) coordinates where row 0 = rank 8, col 0 = file a
    
    Examples:
        >>> notation_to_coords('a1')
        (7, 0)
        
        >>> notation_to_coords('e4')
        (3, 4)
        
        >>> notation_to_coords('h8')
        (0, 7)
    """
    square = square.lower()
    
    if len(square) != 2:
        raise ValueError(f"Invalid square notation: {square}. Expected 2 letters.")
    
    if not square[0].isalpha() or not square[1].isdigit():
        raise ValueError(f"Invalid square notation: {square}. Expected format: [a-h][1-8]")
    
    file_to_col: dict[str, int] = {c: i for i, c in enumerate('abcdefgh')}
    
    ranks: dict[str, int] = {'1': 7, '2': 6, '3': 5, '4': 4,
                            '5': 3, '6': 2, '7': 1, '8': 0}
    
    col = file_to_col[square[0]]
    row = ranks[square[1]]
    
    return row, col


def square_notation(row: int, col: int) -> str:
    """
    Convert array coordinates to algebraic notation.
    
    Args:
        row: Row index (0-7), 0 = rank 8
        col: Column index (0-7), 0 = file a
    
    Returns:
        Algebraic notation (e.g., 'e4', 'a1')
    """
    ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
    files = 'abcdefgh'
    
    return f"{files[col]}{ranks[row]}"


def files() -> list:
    """Return list of files 'a' through 'h'."""
    return list('abcdefgh')


def ranks() -> list:
    """Return list of ranks '1' through '8'."""
    return ['8', '7', '6', '5', '4', '3', '2', '1']
