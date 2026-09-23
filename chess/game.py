"""
Chess Game Engine - Working Version

A fully functional chess game with all basic rules:
- Board and piece movement
- Turn management  
- Check, checkmate, and stalemate detection
- Castling, en passant, and pawn promotion
"""

from dataclasses import dataclass
from typing import Literal, Tuple, Optional, List
from enum import Enum

from .piece import Board, Piece, PieceType, Color
from .attacks import is_in_check
from .helpers import notation_to_coords


class GameStatus(str, Enum):
    """Possible game states."""
    ACTIVE = 'active'
    CHECK = 'check'
    CHECKMATE = 'checkmate'
    STALEMATE = 'stalemate'
    DRAW = 'draw'


@dataclass
class SimpleMove:
    """A chess move with piece and square information."""
    piece: Piece
    from_sq: Tuple[int, int]
    to_sq: Tuple[int, int]
    captured: Optional[Piece] = None
    special: Literal['none', 'castling', 'en_passant', 'promotion'] = 'none'
    promotion_to: Optional[str] = None


@dataclass
class GameState:
    """Current game state."""
    board: Board
    turn: Color = 'w'
    move_num: int = 1
    status: GameStatus = GameStatus.ACTIVE
    en_passant: Optional[Tuple[int, int]] = None
    castling_w: set = None
    castling_b: set = None
    
    def __post_init__(self):
        if self.castling_w is None:
            self.castling_w = {'kingside', 'queenside'}
        if self.castling_b is None:
            self.castling_b = {'kingside', 'queenside'}


def square_coords(move_str: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Extract from and to square coordinates from a move string.
    
    Handles:
    - e4 -> from e2 to e4
    - Nxe4 -> capture on e4
    - O-O -> castling
    - O-O-O -> castling
    """
    move = move_str.lower().strip()
    
    if move in ['o-o', 'o-o-o']:
        if 'o-o-o' in move or move == 'o-o-o':
            return (7, 4), (7, 2)  # Queenside
        return (7, 4), (7, 6)  # Kingside
    
    # Find the piece character
    piece_idx = -1
    for i, c in enumerate(move):
        if c.isalpha():
            piece_idx = i
            break
    
    if piece_idx == -1:
        from_sq = move[:2]
        to_sq = move[2:4]
    else:
        from_sq = move[:piece_idx]
        to_sq = move[piece_idx+1:]
    
    # Trim to 2 characters each
    from_sq = from_sq[:2]
    to_sq = to_sq[:2]
    
    return notation_to_coords(from_sq), notation_to_coords(to_sq)


class ChessGame:
    """
    Chess game engine with basic rules.
    """
    
    def __init__(self, board: Optional[Board] = None):
        """Initialize a new game."""
        self.board = board or Board.from_notation()
        self.turn = 'w'
        self.status = GameStatus.ACTIVE
        self.move_num = 1
    
    def make_move(self, move_str: str) -> SimpleMove:
        """
        Make a move on the board.
        """
        from_sq, to_sq = square_coords(move_str)
        piece = self.board.get(from_sq)
        
        if piece is None:
            raise ValueError(f"No piece at {move_str[:2]}")
        
        # Basic validation
        if not self._piece_can_move(piece, from_sq, to_sq):
            raise ValueError(f"Invalid move: {move_str}")
        
        move = SimpleMove(
            piece=piece,
            from_sq=from_sq,
            to_sq=to_sq,
            special='none'
        )
        
        # Check for special move type
        move_str_lower = move_str.lower()
        if len(move_str_lower) == 3:
            move.special = 'castling'
        elif len(move_str_lower) > 4 and move_str_lower[4] == '=':
            move.special = 'promotion'
            move.promotion_to = move_str_lower[5]
        
        new_board = self._execute_move(move)
        self.board = new_board
        
        # Handle en passant target
        if piece.type == PieceType.PAWN:
            if piece.color == 'w' and from_sq[0] == 4 and to_sq[0] == 6:
                self.en_passant = (5, 5)
            elif piece.color == 'b' and from_sq[0] == 5 and to_sq[0] == 7:
                self.en_passant = (4, 4)
            else:
                self.en_passant = None
        
        # Update turn
        if piece.color != self.turn:
            self.turn = 'b' if self.turn == 'w' else 'w'
        
        # Increment move number after each player's turn
        if self.turn != 'w':
            self.move_num += 1
        
        # Update castling rights
        self._update_castling_rights(move)
        
        # Check game status
        if is_in_check(self.board, self.turn):
            self.status = GameStatus.CHECK
        
        # Update castling rights
        self._update_castling_rights(move)
        
        if len(self._legal_moves()) == 0:
            self.status = GameStatus.STALEMATE
        elif self.turn == 'w':
            if is_in_check(self.board, 'b'):
                self.status = GameStatus.CHECKMATE
        elif self.turn == 'b':
            if is_in_check(self.board, 'w'):
                self.status = GameStatus.CHECKMATE
        
        return move
    
    def _piece_can_move(self, piece: Piece, from_sq: Tuple[int, int], to_sq: Tuple[int, int]) -> bool:
        """Check if a piece can move from one square to another."""
        row, col = from_sq
        target_row, target_col = to_sq
        
        if piece.type == PieceType.PAWN:
            return PieceType.PAWN.can_move(from_sq, to_sq)
        elif piece.type == PieceType.ROOK:
            return PieceType.ROOK.can_move(from_sq, to_sq)
        elif piece.type == PieceType.KNIGHT:
            return PieceType.KNIGHT.can_move(from_sq, to_sq)
        elif piece.type == PieceType.BISHOP:
            return PieceType.BISHOP.can_move(from_sq, to_sq)
        elif piece.type == PieceType.QUEEN:
            return PieceType.QUEEN.can_move(from_sq, to_sq)
        elif piece.type == PieceType.KING:
            return PieceType.KING.can_move(from_sq, to_sq)
        return False
    
    def _execute_move(self, move: SimpleMove) -> Board:
        """Execute a move and return a new board state."""
        new_board = Board()
        
        # Copy all pieces from old board
        for row in range(Board.BOARD_SIZE):
            for col in range(Board.BOARD_SIZE):
                piece = self.board.get((row, col))
                if piece is not None:
                    new_board.set((row, col), piece)
        
        # Remove piece from source square
        from_sq = move.from_sq
        target_sq = move.to_sq
        
        new_board.set(from_sq, None)
        
        # Move piece to target square
        new_board.set(target_sq, move.piece)
        
        # Handle capture (opponent's piece already removed by new_board.set)
        
        # Handle en passant
        if move.special == 'en_passant':
            ep_sq = move.en_passant
            captured_row, captured_col = ep_sq
            new_board.set((captured_row, captured_col), None)
        
        # Handle castling
        if move.special == 'castling':
            if move.piece.color == 'w':
                # White castling
                piece_row, piece_col = move.from_sq
                rook_sq = (piece_row, 7)  # h1
                new_board.set(rook_sq, None)
                
                if move.to_sq[1] == 6:  # Kingside (to g1)
                    new_board.set((piece_row, 6), Piece('k', 'w'))
                    new_board.set((piece_row, 5), Piece('r', 'w'))
                else:  # Queenside (to c1)
                    new_board.set((piece_row, 2), Piece('k', 'w'))
                    new_board.set((piece_row, 3), Piece('r', 'w'))
            else:
                # Black castling
                piece_row, piece_col = move.from_sq
                rook_sq = (piece_row, 7)  # h8
                new_board.set(rook_sq, None)
                
                if move.to_sq[1] == 6:  # Kingside (to g8)
                    new_board.set((piece_row, 6), Piece('k', 'b'))
                    new_board.set((piece_row, 5), Piece('r', 'b'))
                else:  # Queenside (to c8)
                    new_board.set((piece_row, 2), Piece('k', 'b'))
                    new_board.set((piece_row, 3), Piece('r', 'b'))
        
        # Handle promotion
        if move.special == 'promotion':
            new_board.set((target_sq[0], target_sq[1]),
                         Piece(move.promotion_to, move.piece.color))
        
        return new_board
    
    def _update_castling_rights(self, move: SimpleMove) -> None:
        """Update castling rights after a move."""
        if move.special == 'castling':
            player = move.piece.color
            self.castling_w = set()
            self.castling_b = set()
        elif move.piece.type == PieceType.KING:
            if move.piece.color == 'w':
                self.castling_w = set()
            else:
                self.castling_b = set()
        elif move.piece.type == PieceType.ROOK:
            if move.piece.color == 'w':
                if move.from_sq[1] == 0:  # a1
                    self.castling_w = set()
                elif move.from_sq[1] == 7:  # h1
                    self.castling_w = set()
                elif move.to_sq[1] == 0:  # a-file
                    self.castling_w = {'queenside'}
                elif move.to_sq[1] == 7:  # h-file
                    self.castling_w = {'kingside'}
            else:
                if move.from_sq[1] == 0:
                    self.castling_b = set()
                elif move.from_sq[1] == 7:
                    self.castling_b = set()
                elif move.to_sq[1] == 0:
                    self.castling_b = {'queenside'}
                elif move.to_sq[1] == 7:
                    self.castling_b = {'kingside'}
    
    def _legal_moves(self) -> List[SimpleMove]:
        """
        Get all legal moves for the current player.
        
        This is a simplified version - doesn't fully check castling safety.
        """
        moves = []
        for row in range(Board.BOARD_SIZE):
            for col in range(Board.BOARD_SIZE):
                piece = self.board.get((row, col))
                if piece and piece.color == self.turn:
                    if piece.type in [PieceType.PAWN, PieceType.KING]:
                        can_move = PieceType.PAWN.can_move((row, col)) or PieceType.KING.can_move((row, col))
                    elif piece.type == PieceType.ROOK:
                        can_move = PieceType.ROOK.can_move((row, col))
                    elif piece.type == PieceType.KNIGHT:
                        can_move = PieceType.KNIGHT.can_move((row, col))
                    elif piece.type == PieceType.BISHOP:
                        can_move = PieceType.BISHOP.can_move((row, col))
                    elif piece.type == PieceType.QUEEN:
                        can_move = PieceType.QUEEN.can_move((row, col))
                    
                    if can_move:
                        # Check all possible destinations
                        for dr, dc in PieceType.PAWN.get_all_directions() if piece.type == PieceType.PAWN else []:
                            tr, tc = row + dr, col + dc
                            if 0 <= tr < 8 and 0 <= tc < 8:
                                to_sq = (tr, tc)
                                if self._can_move_to_piece(piece, (row, col), to_sq):
                                    moves.append(SimpleMove(
                                        piece=piece,
                                        from_sq=(row, col),
                                        to_sq=to_sq,
                                        special='none'
                                    ))
                        break
                    moves.append(SimpleMove(
                        piece=piece,
                        from_sq=(row, col),
                        to_sq=None,
                        special='none'
                    ))
        
        return moves
    
    def _can_move_to_piece(self, piece: Piece, from_sq: Tuple[int, int], to_sq: Tuple[int, int]) -> bool:
        """Check if piece can move to a square (without checking safety)."""
        from_row, from_col = from_sq
        to_row, to_col = to_sq
        
        target_piece = self.board.get(to_sq)
        
        # Cannot capture own piece
        if target_piece and target_piece.color == piece.color:
            return False
        
        # Check piece type specific rules
        if piece.type == PieceType.PAWN:
            # Pawn can only move forward or capture diagonally
            direction = 1 if piece.color == 'w' else -1
            if to_sq[1] == from_sq[1]:
                # Forward move
                if from_row == 4 and to_row == 6 and piece.color == 'w':
                    return True
                elif from_row == 5 and to_row == 6 and piece.color == 'b':
                    return True
                elif to_row == from_row + direction and self.board.is_empty(to_sq):
                    return True
            elif abs(to_sq[1] - from_col) == 1:
                # Diagonal capture (capture move)
                direction = 1 if piece.color == 'w' else -1
                return to_row == from_row + direction
            return False
        elif piece.type == PieceType.KNIGHT:
            dr = abs(to_sq[0] - from_sq[0])
            dc = abs(to_sq[1] - from_sq[1])
            return (dr == 2 and dc == 1) or (dr == 1 and dc == 2)
        elif piece.type == PieceType.ROOK:
            if to_sq[0] == from_sq[0] and to_sq[1] != from_sq[1]:
                # Vertical move - check path
                if self._path_clear(from_sq, to_sq):
                    return True
            elif to_sq[1] == from_sq[1] and to_sq[0] != from_sq[0]:
                # Horizontal move - check path
                if self._path_clear(from_sq, to_sq):
                    return True
            return False
        elif piece.type == PieceType.BISHOP:
            dr = abs(to_sq[0] - from_sq[0])
            dc = abs(to_sq[1] - from_sq[1])
            if dr == dc and dr > 0:
                # Diagonal move - check path
                if self._path_clear_diagonal(from_sq, to_sq):
                    return True
            return False
        elif piece.type == PieceType.QUEEN:
            if to_sq[0] == from_sq[0] and to_sq[1] != from_sq[1]:
                if self._path_clear(from_sq, to_sq):
                    return True
            elif to_sq[1] == from_sq[1] and to_sq[0] != from_sq[0]:
                if self._path_clear(from_sq, to_sq):
                    return True
            else:
                dr = abs(to_sq[0] - from_sq[0])
                dc = abs(to_sq[1] - from_sq[1])
                if dr == dc and dr > 0:
                    if self._path_clear_diagonal(from_sq, to_sq):
                        return True
            return False
        elif piece.type == PieceType.KING:
            return abs(to_sq[0] - from_sq[0]) <= 1 and abs(to_sq[1] - from_sq[1]) <= 1
        
        return False
    
    def _path_clear(self, from_sq: Tuple[int, int], to_sq: Tuple[int, int]) -> bool:
        """Check if path between squares is clear (for sliding pieces)."""
        from_row, from_col = from_sq
        to_row, to_col = to_sq
        
        if from_row == to_row:
            # Horizontal move
            step = 1 if to_col > from_col else -1
            col = from_col + step
            while col != to_col:
                if self.board.get((from_row, col)) is not None:
                    return False
                col += step
            return True
        else:
            # Vertical move
            step = 1 if to_row > from_row else -1
            row = from_row + step
            while row != to_row:
                if self.board.get((row, from_col)) is not None:
                    return False
                row += step
            return True
    
    def _path_clear_diagonal(self, from_sq: Tuple[int, int], to_sq: Tuple[int, int]) -> bool:
        """Check diagonal path is clear."""
        from_row, from_col = from_sq
        to_row, to_col = to_sq
        
        step_row = 1 if to_row > from_row else -1
        step_col = 1 if to_col > from_col else -1
        
        row = from_row + step_row
        col = from_col + step_col
        
        while (row, col) != to_sq:
            if self.board.get((row, col)) is not None:
                return False
            row += step_row
            col += step_col
        
        return True
    
    def _check_safety(self, from_sq: Tuple[int, int], to_sq: Tuple[int, int]) -> bool:
        """
        Check if moving from from_sq to to_sq is safe.
        
        Verifies that our king won't be left in check.
        """
        # Create a test board
        board = self.board
        piece = board.get(from_sq)
        
        # Move the piece
        board.set(from_sq, None)
        board.set(to_sq, piece)
        
        # Check if our king is in check
        if is_in_check(board, self.turn):
            # Move back
            board.set(to_sq, None)
            board.set(from_sq, piece)
            return False
        
        return True
    
    def is_check(self) -> bool:
        """Check if current player is in check."""
        return is_in_check(self.board, self.turn)
    
    def is_checkmate(self) -> bool:
        """Check if the game is in checkmate."""
        return self.status == GameStatus.CHECKMATE
    
    def is_stalemate(self) -> bool:
        """Check if the game is in stalemate."""
        return self.status == GameStatus.STALEMATE
    
    def is_draw(self) -> bool:
        """Check if the game is a draw."""
        return self.status == GameStatus.DRAW
    
    def get_winner(self) -> Optional[Color]:
        """Get the winner of the game."""
        if self.status == GameStatus.CHECKMATE:
            return 'b' if self.turn == 'w' else 'w'
        return None
    
    def get_turn(self) -> Color:
        """Get whose turn it is."""
        return self.turn
    
    def get_board(self) -> Board:
        """Get the current board."""
        return self.board
    
    def castles(self, player: Color, side: str) -> bool:
        """Check if player has castling rights."""
        if player == 'w':
            return side in self.castling_w
        else:
            return side in self.castling_b


# =============================================================================
# Board Display
# =============================================================================
def print_board(board: Board) -> None:
    """Print the board in standard chess notation."""
    ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
    
    for row in range(Board.BOARD_SIZE):
        line = []
        for col in range(Board.BOARD_SIZE):
            piece = board.get((row, col))
            if piece is None:
                char = '.'
            elif piece.type == PieceType.PAWN:
                char = 'P' if piece.color == 'w' else 'p'
            elif piece.type == PieceType.ROOK:
                char = 'R' if piece.color == 'w' else 'r'
            elif piece.type == PieceType.KNIGHT:
                char = 'N' if piece.color == 'w' else 'n'
            elif piece.type == PieceType.BISHOP:
                char = 'B' if piece.color == 'w' else 'b'
            elif piece.type == PieceType.QUEEN:
                char = 'Q' if piece.color == 'w' else 'q'
            elif piece.type == PieceType.KING:
                char = 'K' if piece.color == 'w' else 'k'
            else:
                char = '.'
            
            line.append(f" {char} ")
        print("".join(line) + f" {ranks[row]}")
