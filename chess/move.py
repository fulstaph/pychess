"""
Chess Move Module

Concept: A move describes transitioning from one board state to another.
It includes the piece moved, from square, to square, and any special flags
(capture, promotion, en passant, castling).

Key concepts:
1. Move validation - not all moves are legal
2. Move generation - generating all possible moves for a position
3. Move execution - applying a move to create new board state
4. Special moves - castling, en passant, pawn promotion
"""

from dataclasses import dataclass, field
from typing import Literal
from enum import Enum

from .piece import Piece, PieceType, Board, Square, Color


# =============================================================================
# Move Notation Types
# =============================================================================
# Standard move notation: "e2e4" means move piece from e2 to e4
# Capture notation: "e2xe4" means capture on e4
# Disambiguated moves: "Nge7" means knight from g-file to e7
MoveString = str
CastlingType = Literal['K', 'Q', 'k', 'q']  # K=white kingside, Q=white queenside, etc.


# =============================================================================
# Piece Move Directions
# =============================================================================
# Represents possible direction(s) a piece can move on the board
# Each tuple is (delta_row, delta_col)
class MoveDirection:
    """
    Represents direction(s) a piece can move.
    
    Example:
        PawnWhite = MoveDirection([(1, 0)])  # White pawn moves forward (down in array)
        KnightAny = MoveDirection([(2, 0), (-2, 0), (0, 2), (0, -2), ...])  # Knight L-shape
    """
    
    def __init__(self, directions: list[tuple[int, int]] = None) -> None:
        """Initialize with list of (row_delta, col_delta) tuples."""
        self.directions = directions or []
    
    def is_valid_move(self, from_square: Square, to_square: Square) -> bool:
        """Check if to_square is in the valid direction from from_square."""
        row_diff = to_square[0] - from_square[0]
        col_diff = to_square[1] - from_square[1]
        
        # Check if direction matches
        return (row_diff, col_diff) in self.directions


# =============================================================================
# Basic Move Representation
# =============================================================================
@dataclass
class Move:
    """
    Represents a single chess move.
    
    A move consists of:
    - piece: The piece being moved
    - from_square: Starting square
    - to_square: Destination square
    - captured_piece: The piece captured (if any), otherwise None
    - special: Any special move type (castling, promotion)
    """
    piece: Piece
    from_square: Square
    to_square: Square
    captured_piece: Piece | None = None
    special: Literal['none', 'castle_k', 'castle_q', 'en_passant', 'promotion'] = 'none'
    promotion_to: PieceType | None = None
    
    def is_capture(self) -> bool:
        """Check if this move is a capture (takes opponent's piece)."""
        return self.captured_piece is not None
    
    def is_legal(self, board: Board) -> bool:
        """
        Check if this move is legal according to chess rules.
        
        Legal moves must:
        1. Move piece according to its movement rules
        2. Not pass through occupied squares (except knights)
        3. Land on empty square or capture opponent piece
        4. Not leave/put own king in check
        5. Not involve moving through check (unless king or capturing check)
        6. Not involve moving through own pieces
        7. Not involve capturing own pieces
        
        Note: This is a simplified check - some edge cases may not be handled
        (like en passant validation, king safety check on intermediate squares)
        """
        return self._check_piece_can_move(board, from_square, to_square) and \
               self._check_king_not_in_check_after(board, to_square, from_square) and \
               self._check_castling_valid(board) and \
               self._check_en_passant_valid()
    
    def execute(self, board: Board) -> 'Board':
        """
        Execute this move, creating a new board state.
        
        Args:
            board: The current board state
            
        Returns:
            New board state after executing the move
        """
        new_board = board  # Copy by reference (we'll modify in place conceptually)
        
        # For simplicity in our minimal engine, we'll just modify in place
        # and create a new board copy
        return self._apply(board)
    
    def _apply(self, board: Board) -> 'Board':
        """Apply move to board, return modified board."""
        new_board = Board()
        
        # Copy all pieces from old board
        for row in range(Board.BOARD_SIZE):
            for col in range(Board.BOARD_SIZE):
                piece = board.get((row, col))
                if piece is not None:
                    new_board.set((row, col), piece)
        
        # Apply the move
        self._apply_move(new_board)
        
        return new_board
    
    def _apply_move(self, board: Board) -> None:
        """Apply the move on the board (in-place)."""
        row, col = board.to_coords(self.from_square)
        target_row, target_col = board.to_coords(self.to_square)
        
        piece = board.get((row, col))
        target_piece = board.get((target_row, target_col))
        
        # 1. Move the piece
        board.set((row, col), None)  # Remove from original square
        board.set((target_row, target_col), piece)  # Place on target square
        
        # 2. Handle captures (opponent's piece removed automatically by removing piece)
        
        # 3. Handle special moves
        if self.special == 'promotion':
            # Promote pawn to specified type (usually queen)
            piece = Piece(self.promotion_to or PieceType.QUEEN, piece.color)
            board.set((target_row, target_col), piece)
        
        if self.special == 'en_passant':
            # En passant: remove the pawn that was just captured
            # After white moves e5-e4, black pawn is on e6
            ep_row, ep_col = target_row - 1, target_col
            board.set((ep_row, ep_col), None)
        
        if self.special == 'castle_k':
            # Kingside castling: King moves 2 squares, rook jumps
            # King from e1 to g1 (row 0, col 4 -> row 0, col 6)
            king_row, king_col = 0, 4
            rook_row, rook_col = 0, 7
            
            # King moves 2 squares (skip one square)
            board.set((king_row, king_col), None)
            board.set((king_row, king_col + 2), Piece('k', Piece('k', 'w').color))
            
            # Rook jumps from h-file to f-file
            board.set((rook_row, rook_col), None)
            board.set((rook_row, rook_col - 1), Piece('r', Piece('r', 'w').color))
        
        if self.special == 'castle_q':
            # Queenside castling: King moves 2 squares, rook jumps
            # King from e1 to c1 (row 0, col 4 -> row 0, col 2)
            king_row, king_col = 0, 4
            rook_row, rook_col = 0, 0
            
            board.set((king_row, king_col), None)
            board.set((king_row, king_col - 2), Piece('k', Piece('k', 'w').color))
            
            # Rook jumps from a-file to d-file
            board.set((rook_row, rook_col), None)
            board.set((rook_row, rook_col + 1), Piece('r', Piece('r', 'w').color))


# =============================================================================
# Move Generation Helpers
# =============================================================================
def generate_possible_moves(piece: Piece, from_square: Square, to_square: Square) -> bool:
    """
    Check if a piece can legally move to a square (ignoring check constraints).
    
    This generates all valid moves for a given piece to all valid squares.
    Called for each piece on the board to generate all possible moves.
    
    Note: This is a "pseudo-legal" move generator - doesn't check if king
    would be in check after the move.
    """
    # Simple direction checking for each piece type
    # In a more complete engine, we'd generate actual moves step by step
    
    return True  # Simplified - actual implementation would check directions


# =============================================================================
# Move Notation Generator
# =============================================================================
def generate_move_notation(from_square: Square, to_square: Square, 
                           captured_piece: Piece | None = None,
                           move_string: MoveString | None = None) -> str:
    """
    Generate algebraic notation for a move.
    
    Examples:
        - 'e2e4' for pawn moving e2 to e4
        - 'exe4' or 'Nxe4' for capturing on e4
        - 'O-O' for kingside castling
        - 'O-O-O' for queenside castling
        - 'e7-e8=Q' for en passant or promotion
    """
    if move_string is not None:
        return move_string
    
    piece_type = from_square[1] if from_square[0] == to_square[0] else 1  # Simplified
    piece_type_char = 'N'  # Default
    
    # Determine the piece character
    chars = {'p': 'P', 'r': 'R', 'n': 'n', 'b': 'B', 'q': 'Q', 'k': 'K'}
    piece_type_char = chars.get(from_square[1], 'P')
    
    # Handle castling
    if captured_piece is None and abs(from_square[1] - to_square[1]) == 2:
        if from_square[0] == to_square[0] == 0:  # White's rank
            return 'O-O' if to_square[1] == 6 else 'O-O-O'
        else:  # Black's rank
            return 'o-o' if to_square[1] == 6 else 'o-o-o'
    
    # Need to disambiguate if multiple pieces of same type can move here
    notation = f"{piece_type_char}{file_letter(from_square[1])}{file_letter(to_square[1])}{rank_letter(to_square[0])}"
    
    if captured_piece is not None:
        # Add 'x' for capture (between pieces, or just 'x' for pawns)
        if piece_type_char == 'P':
            notation = f"{rank_letter(from_square[0])}x{notation}"
        else:
            notation = f"{piece_type_char}x{notation}"
    
    return notation


def file_letter(col: int) -> str:
    """Convert column (0-7) to file letter ('a'-'h')."""
    return chr(ord('a') + col)


def rank_letter(row: int) -> str:
    """Convert row (0-7) to rank number ('1'-'8')."""
    return str(8 - row)  # Array row 0 = rank 8


# =============================================================================
# Game State and Result
# =============================================================================
class GameResult(str, Enum):
    """Possible end game results."""
    WHITE_CHECKMATE = 'White wins by checkmate'
    BLACK_CHECKMATE = 'Black wins by checkmate'
    DRAW = 'Draw'
    WHITE_FIFTH_IN_LINE = 'Draw by 50-move rule (white)'
    BLACK_FIFTH_IN_LINE = 'Draw by 50-move rule (black)'
    STALEMATE = 'Stalemate'
    INSUFFICIENT_MATERIAL = 'Insufficient material (draw)'
    THREEFOLD_REPETITION = 'Threefold repetition'
    FIFTY_MOVE_RULE = '50-move rule'
    SEVENTY_MOVE_RULE = '70-move rule'


def generate_possible_moves_from_position(board: Board) -> list[Move]:
    """
    Generate all pseudo-legal moves from a given board position.
    
    This generates all moves that:
    1. Are valid according to piece movement rules
    2. Don't pass through enemy pieces (except knights)
    3. Land on empty squares or capture enemy pieces
    4. Don't capture own pieces
    
    Note: Does NOT check if king would be in check after move.
    This function generates all pseudo-legal moves; we need additional
    validation to check for king safety.
    """
    moves: list[Move] = []
    
    # For each square on the board
    for row in range(Board.BOARD_SIZE):
        for col in range(Board.BOARD_SIZE):
            piece = board.get((row, col))
            
            # Only consider pieces that belong to the player who can move
            if piece is None:
                continue
            
            piece_can_move = piece_can_move_to(piece, (row, col), board)
            
            if not piece_can_move:
                continue
            
            # Generate all valid target squares for this piece
            for dr, dc in piece.get_moves():
                target_row = row + dr
                target_col = col + dc
                
                if not (0 <= target_row < Board.BOARD_SIZE and 0 <= target_col < Board.BOARD_SIZE):
                    continue
                
                target_square = (target_row, target_col)
                
                # Check if move is valid for this piece type
                if not piece.get_move_direction().is_valid_move((row, col), target_square):
                    continue
                
                # Apply piece-specific movement rules
                valid_target = piece.get_valid_targets(row, col, target_square, board)
                
                if valid_target:
                    moves.append(Move(
                        piece=piece,
                        from_square=(row, col),
                        to_square=target_square
                    ))
    
    return moves


def piece_can_move_to(piece: Piece, from_square: Square, board: Board) -> bool:
    """
    Check if a piece can legally move to a square.
    
    Returns True if:
    - Piece is not blocked from moving in the right direction
    - Target square is empty or has opponent's piece
    - Move doesn't move through own pieces
    """
    # Check target square is not own piece (basic check)
    target_piece = board.get(from_square)
    if target_piece is not None:
        return False
    
    return True


# =============================================================================
# Piece Movement Generators
# =============================================================================
def pawn_moves(board: Board, piece: Piece) -> list[tuple[int, int]]:
    """
    Generate all valid moves for a pawn.
    
    Pawn rules:
    1. Move forward 1 square (2 squares on first move if no capture)
    2. Capture diagonally forward 1 square
    3. En passant capture (special case)
    4. Can't move through own pieces
    5. Can't capture own pieces
    
    Returns list of (delta_row, delta_col) directions.
    """
    moves: list[tuple[int, int]] = []
    
    if piece.color == 'w':  # White pawns move "down" in array (rank 7 to rank 0)
        forward = (1, 0)
        backward = (-1, 0)
        diag_left = (1, -1)
        diag_right = (1, 1)
        
        from_row, from_col = 7, piece.type  # White pawns start on rank 7
    else:  # Black pawns move "up" in array (rank 0 to rank 7)
        forward = (-1, 0)
        backward = (1, 0)
        diag_left = (-1, -1)
        diag_right = (-1, 1)
    
    return moves


def rook_moves(board: Board, piece: Piece) -> list[tuple[int, int]]:
    """
    Generate all valid moves for a rook.
    
    Rook moves in 4 directions (straight lines):
    - Up (negative row)
    - Down (positive row)
    - Left (negative col)
    - Right (positive col)
    
    Sliding piece: can move any distance in valid direction
    """
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
    return moves


def knight_moves(board: Board, piece: Piece) -> list[tuple[int, int]]:
    """
    Generate all valid moves for a knight.
    
    Knight moves in L-shape (2+1 squares):
    - 8 possible directions, can jump over pieces
    
    Knight direction offsets:
    (±1, ±2) and (±2, ±1)
    """
    moves = [
        (-2, -1), (-2, 1),
        (-1, -2), (-1, 2),
        (1, -2), (1, 2),
        (2, -1), (2, 1)
    ]
    return moves


def bishop_moves(board: Board, piece: Piece) -> list[tuple[int, int]]:
    """
    Generate all valid moves for a bishop.
    
    Bishop moves diagonally in 4 directions:
    - Up-left (-1, -1)
    - Up-right (-1, 1)
    - Down-left (1, -1)
    - Down-right (1, 1)
    
    Sliding piece: can move any distance diagonally
    """
    moves = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal directions
    return moves


def queen_moves(board: Board, piece: Piece) -> list[tuple[int, int]]:
    """
    Generate all valid moves for a queen.
    
    Queen combines rook and bishop movement:
    - 4 orthogonal directions (like rook)
    - 4 diagonal directions (like bishop)
    """
    return rook_moves(board, piece) + bishop_moves(board, piece)


def king_moves(board: Board, piece: Piece) -> list[tuple[int, int]]:
    """
    Generate all valid moves for a king.
    
    King moves 1 square in any direction:
    - 8 surrounding squares
    
    King rules:
    - Cannot move into check
    - Cannot move through check
    """
    moves = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    return moves


# =============================================================================
# Piece Movement Implementation
# =============================================================================
def get_piece_moves(board: Board, piece: Piece, square: Square) -> list[Move]:
    """
    Get all valid moves for a piece at a given square.
    
    Args:
        board: Current board state
        piece: The piece whose moves we're generating
        square: The square where the piece currently is
    
    Returns:
        List of valid Move objects for this piece
    """
    moves: list[Move] = []
    
    row, col = square
    
    # Get direction type based on piece
    if piece.type == PieceType.PAWN:
        directions = pawn_moves(board, piece)
    elif piece.type == PieceType.ROOK:
        directions = rook_moves(board, piece)
    elif piece.type == PieceType.KNIGHT:
        directions = knight_moves(board, piece)
    elif piece.type == PieceType.BISHOP:
        directions = bishop_moves(board, piece)
    elif piece.type == PieceType.QUEEN:
        directions = queen_moves(board, piece)
    elif piece.type == PieceType.KING:
        directions = king_moves(board, piece)
    else:
        return moves
    
    # Generate moves in each direction
    for dr, dc in directions:
        target_row, target_col = row + dr, col + dc
        
        # Check bounds
        if not (0 <= target_row < Board.BOARD_SIZE and 0 <= target_col < Board.BOARD_SIZE):
            continue
        
        target_square = (target_row, target_col)
        
        # Check piece blocking (except for knights)
        if piece.type == PieceType.KNIGHT:
            # Knight can jump
            valid = piece_can_move_to(piece, square, target_square)
        else:
            # Sliding pieces (rook, bishop, queen, pawn)
            valid = slide_moves(board, piece, square, target_square)
        
        if valid:
            moves.append(Move(
                piece=piece,
                from_square=square,
                to_square=target_square
            ))
    
    return moves


def piece_can_move_to(piece: Piece, from_square: Square, to_square: Square) -> bool:
    """
    Check if piece can move from one square to another.
    
    Args:
        piece: The piece attempting to move
        from_square: Starting square (piece's current location)
        to_square: Target square
    
    Returns:
        True if move is valid
    """
    # Check if we're landing on own piece
    target_piece = Board.get_board().get(to_square)
    if target_piece is not None and target_piece.color == piece.color:
        return False
    
    return True


def slide_moves(board: Board, piece: Piece, from_square: Square, to_square: Square) -> bool:
    """
    Check if sliding piece can move from one square to another.
    
    Sliding pieces (rook, bishop, queen) must have a clear path.
    """
    dr = to_square[0] - from_square[0]
    dc = to_square[1] - from_square[1]
    
    # Determine movement type
    is_vertical = dc == 0
    is_horizontal = dr == 0
    
    # Normalize direction (remove sign, step)
    step = 0
    if is_vertical:
        step = -1 if dr < 0 else 1
    elif is_horizontal:
        step = -1 if dc < 0 else 1
    
    # Check each square along the path
    current = from_square
    step_row = step * to_square[0] // (abs(to_square[0] - from_square[0]) + 1) if dr != 0 else 0
    step_col = step * to_square[1] // (abs(to_square[1] - from_square[1]) + 1) if dc != 0 else 0
    
    row, col = current
    max_row, max_col = to_square
    
    while (row, col) != to_square:
        # Check if any sliding piece blocks path
        if any_piece_blocks_path((row, col), to_square, board, piece):
            return False
        
        # Move to next square in direction
        if is_vertical:
            row += step
        else:
            col += step
        current = (row, col)
    
    return True


def any_piece_blocks_path(from_square: Square, to_square: Square, board: Board, 
                         my_color: Color) -> bool:
    """
    Check if any pieces of either color block the path.
    
    For sliding pieces, we need to check that no pieces (friend or foe)
    block the movement path, except at the destination (capture).
    """
    from_row, from_col = from_square
    to_row, to_col = to_square
    
    # Determine direction
    is_vertical = from_col == to_col
    is_horizontal = from_row == to_row
    
    if not is_vertical and not is_horizontal:
        return False  # Can't move diagonally if not same diagonal
    
    step = 0
    if is_vertical:
        step = -1 if from_row > to_row else 1
    else:
        step = -1 if from_col > to_col else 1
    
    # Traverse path
    row, col = from_row, from_col
    cur_row, cur_col = to_row, to_col
    
    while (row, col) != (cur_row, cur_col):
        if board.get((row, col)) is not None:
            return True  # Path is blocked
        row += step
        col += step
    
    return False


# =============================================================================
# Castling Validation
# =============================================================================
def can_castle_king_side(board: Board, color: Color) -> bool:
    """
    Check if white can castle kingside.
    
    Requirements:
    1. King is on e1, rook is on h1 (or has moved before)
    2. King hasn't moved before
    3. Rook on h-file hasn't moved
    4. King can move to f1 and g1
    5. King is not currently in check
    6. Squares f1 and g1 are not under attack
    7. King doesn't pass through check (on move from e1 to g1)
    """
    return True  # Simplified - full implementation would track king and rook movement


def can_castle_queen_side(board: Board, color: Color) -> bool:
    """
    Check if white can castle queenside.
    
    Requirements:
    1. King is on e1, rook is on a1
    2. King hasn't moved, rook on a-file hasn't moved
    3. King can move to c1 and b1
    4. King is not in check
    5. Squares b1, c1, and d1 are not under attack
    6. King doesn't pass through check
    """
    return True  # Simplified
# =============================================================================
# Castling Move Generation
# =============================================================================
def castling_moves(player: Color, board: Board) -> list[Move]:
    """
    Generate all possible castling moves for a player.
    
    Args:
        player: 'w' for white, 'b' for black
        board: Current board state
    
    Returns:
        List of Move objects for castling
    
    Examples:
        # Get white's possible castling moves
        white_castles = castling_moves('w', board)
    """
    moves = []
    
    # White can castle kingside if:
    # - King is on e1 (row 0, col 4)
    # - King can move to g1 (row 0, col 6)
    if board.get((0, 4)) is not None and \
       board.get((0, 4)).type == PieceType.KING and \
       board.get((0, 7)) is not None and \
       board.get((0, 7)).type == PieceType.ROOK:
        moves.append(Move(
            piece=Piece('k', 'w'),
            from_square=(0, 4),
            to_square=(0, 6),
            special='castling',
            castling='k',
            promotion_to=None,
            captured_piece=None
        ))
    
    # White can castle queenside if:
    # - King is on e1
    # - King can move to c1 (row 0, col 2)
    if board.get((0, 4)) is not None and \
       board.get((0, 4)).type == PieceType.KING and \
       board.get((0, 0)) is not None and \
       board.get((0, 0)).type == PieceType.ROOK:
        moves.append(Move(
            piece=Piece('k', 'w'),
            from_square=(0, 4),
            to_square=(0, 2),
            special='castling',
            castling='q',
            promotion_to=None,
            captured_piece=None
        ))
    
    # Black can castle kingside if:
    # - King is on e8 (row 7, col 4)
    # - King can move to g8 (row 7, col 6)
    if board.get((7, 4)) is not None and \
       board.get((7, 4)).type == PieceType.KING and \
       board.get((7, 7)) is not None and \
       board.get((7, 7)).type == PieceType.ROOK:
        moves.append(Move(
            piece=Piece('k', 'b'),
            from_square=(7, 4),
            to_square=(7, 6),
            special='castling',
            castling='k',
            promotion_to=None,
            captured_piece=None
        ))
    
    # Black can castle queenside if:
    # - King is on e8
    # - King can move to c8 (row 7, col 2)
    if board.get((7, 4)) is not None and \
       board.get((7, 4)).type == PieceType.KING and \
       board.get((7, 0)) is not None and \
       board.get((7, 0)).type == PieceType.ROOK:
        moves.append(Move(
            piece=Piece('k', 'b'),
            from_square=(7, 4),
            to_square=(7, 2),
            special='castling',
            castling='q',
            promotion_to=None,
            captured_piece=None
        ))
    
    return moves
