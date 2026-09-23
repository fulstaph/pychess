"""
Chess AI Engine - Generates LEGAL moves (not random)

Usage: python3 chess/simple_engine_final.py
"""

from typing import Literal, Tuple, Optional, List, Dict
from chess import ChessGame, Board, Piece, PieceType, notation_to_coords, print_board
from chess.piece import Color


class ChessEngine:
    """
    A simple chess AI that generates legal moves.
    """
    
    # Material values for evaluating moves
    PIECE_VALUES = {
        PieceType.PAWN: 100,
        PieceType.KNIGHT: 320,
        PieceType.BISHOP: 330,
        PieceType.ROOK: 500,
        PieceType.QUEEN: 900,
        PieceType.KING: 0,
    }
    
    def __init__(self, board: Board, player: Literal['w', 'b']):
        self.board = board
        self.player = player
    
    def get_all_legal_moves(self) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """
        Generate all legal moves for the current player.
        
        Returns list of (from_sq, to_sq, piece) tuples.
        """
        moves = []
        
        for row in range(8):
            for col in range(8):
                piece = self.board.get((row, col))
                
                if piece is None:
                    continue
                
                if piece.color != self.player:
                    continue
                
                # Generate moves for this piece
                possible_moves = self._get_moves_for_piece(piece, (row, col))
                
                for from_sq, to_sq, piece_name in possible_moves:
                    # Validate move
                    if not self._validate_move(piece, from_sq, to_sq, self.board):
                        continue
                    
                    moves.append((from_sq, to_sq, piece_name))
        
        return moves
    
    def _get_moves_for_piece(self, piece: Piece, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """Get all possible moves for a piece."""
        moves = []
        p_row, p_col = pos
        
        if piece.type == PieceType.PAWN:
            # Pawn moves forward
            direction = 1 if piece.color == 'w' else -1
            start_row = 7 if piece.color == 'w' else 0
            
            # Check forward moves
            new_row = p_row + direction
            if in_bounds(new_row, p_col):
                if self.board.get((new_row, p_col)) is None:
                    moves.append((pos, (new_row, p_col), 'pawn'))
                    
                    # Can also move 2 squares on first move
                    if p_row == start_row:
                        double_row = p_row + 2 * direction
                        if in_bounds(double_row, p_col) and self.board.get((double_row, p_col)) is None:
                            moves.append((pos, (double_row, p_col), 'pawn'))
            
            # Diagonal captures
            for dc in [-1, 1]:
                new_row = p_row + direction
                new_col = p_col + dc
                if in_bounds(new_row, new_col):
                    target = (new_row, new_col)
                    if self.board.get(target) is not None:
                        target_piece = self.board.get(target)
                        if target_piece.color != piece.color:
                            moves.append((pos, target, 'pawn'))
        
        elif piece.type == PieceType.ROOK:
            # Rook moves in straight lines
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_row, new_col = p_row + dr, p_col + dc
                while in_bounds(new_row, new_col):
                    target = (new_row, new_col)
                    target_piece = self.board.get(target)
                    
                    if target_piece is None:
                        moves.append((pos, target, 'rook'))
                    else:
                        if target_piece.color != piece.color:
                            moves.append((pos, target, 'rook'))
                        break
                    
                    new_row += dr
                    new_col += dc
        
        elif piece.type == PieceType.KNIGHT:
            # Knight moves in L-shapes
            knight_offsets = [
                (2, 1), (2, -1), (-2, 1), (-2, -1),
                (1, 2), (1, -2), (-1, 2), (-1, -2)
            ]
            for dr, dc in knight_offsets:
                new_row, new_col = p_row + dr, p_col + dc
                if in_bounds(new_row, new_col):
                    target = (new_row, new_col)
                    target_piece = self.board.get(target)
                    if target_piece is None or target_piece.color != piece.color:
                        moves.append((pos, target, 'knight'))
        
        elif piece.type == PieceType.BISHOP:
            # Bishop moves diagonally
            for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                new_row, new_col = p_row + dr, p_col + dc
                while in_bounds(new_row, new_col):
                    target = (new_row, new_col)
                    target_piece = self.board.get(target)
                    
                    if target_piece is None:
                        moves.append((pos, target, 'bishop'))
                    else:
                        if target_piece.color != piece.color:
                            moves.append((pos, target, 'bishop'))
                        break
                    
                    new_row += dr
                    new_col += dc
        
        elif piece.type == PieceType.QUEEN:
            # Queen moves like rook OR bishop
            rook_moves = self._get_moves_for_piece(Piece(PieceType.ROOK, piece.color), pos)
            bishop_moves = self._get_moves_for_piece(Piece(PieceType.BISHOP, piece.color), pos)
            moves = [(pos, m[1], 'queen') for m in rook_moves if (pos, m[1], 'queen') not in moves]
            moves = moves + [(pos, m[1], 'queen') for m in bishop_moves]
        
        elif piece.type == PieceType.KING:
            # King moves 1 square in any direction
            king_moves = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    new_row, new_col = p_row + dr, p_col + dc
                    if in_bounds(new_row, new_col):
                        target = (new_row, new_col)
                        if self.board.get(target) is None or self.board.get(target).color != piece.color:
                            king_moves.append((pos, target, 'king'))
            moves = king_moves
        
        return moves
    
    def _validate_move(self, piece: Piece, from_sq: Tuple[int, int], to_sq: Tuple[int, int], board: Board) -> bool:
        """
        Check if a move is legal.
        
        A move is legal if:
        1. The piece can move to the target square (movement rules)
        2. The destination square is not under attack by opponent
        3. The move doesn't leave our king in check
        """
        fr, fc = from_sq
        tr, tc = to_sq
        
        # Check 2: Destination must not be under attack
        if is_square_attacked((tr, tc), self.player, board):
            return False
        
        # Check 1: Verify piece can actually move to destination
        if not self._piece_can_move(piece, from_sq, to_sq):
            return False
        
        # Check 3: Make move and verify our king is not in check
        test_board = board.copy()
        
        # Apply the move temporarily
        test_board.set(from_sq, None)
        test_board.set(to_sq, piece)
        
        # Check if our king would be in check
        king_pos = self._find_king(board, self.player)
        if king_pos is not None:
            if is_square_attacked(king_pos, self.player, test_board):
                return False
        
        return True
    
    def _piece_can_move(self, piece: Piece, from_sq: Tuple[int, int], to_sq: Tuple[int, int]) -> bool:
        """Check if a piece can move from one square to another."""
        fr, fc = from_sq
        tr, tc = to_sq
        
        if piece.type == PieceType.PAWN:
            direction = 1 if piece.color == 'w' else -1
            start_row = 7 if piece.color == 'w' else 0
            
            if fr == tr:
                # Straight forward move
                if piece.color == 'w':
                    if tr == 6 and from_sq == (0, fc):
                        return True
                    if tr == 5 and from_sq == (1, fc):
                        return True
                else:
                    if tr == 4 and from_sq == (6, fc):
                        return True
                    if tr == 5 and from_sq == (5, fc):
                        return True
            elif abs(tr - fr) == 1:
                # Diagonal capture
                return True
            return False
        
        elif piece.type == PieceType.ROOK:
            if fr != tr and fc != tc:
                return False
            return is_path_clear(from_sq, to_sq, board)
        
        elif piece.type == PieceType.KNIGHT:
            dr = abs(tr - fr)
            dc = abs(tc - fc)
            return (dr == 2 and dc == 1) or (dr == 1 and dc == 2)
        
        elif piece.type == PieceType.BISHOP:
            dr = abs(tr - fr)
            dc = abs(tc - fc)
            if dr != dc or dr == 0:
                return False
            return is_path_clear(from_sq, to_sq, board)
        
        elif piece.type == PieceType.QUEEN:
            if fr == tr:
                return is_path_clear(from_sq, to_sq, board)
            elif fc == tc:
                return is_path_clear(from_sq, to_sq, board)
            else:
                dr = abs(tr - fr)
                dc = abs(tc - fc)
                return dr == dc and dr > 0 and is_path_clear_diagonal(from_sq, to_sq, board)
        
        elif piece.type == PieceType.KING:
            dr = abs(tr - fr)
            dc = abs(tc - fc)
            return dr <= 1 and dc <= 1
        
        return False
    
    def _find_king(self, board: Board, player: Literal['w', 'b']) -> Optional[Tuple[int, int]]:
        """Find the king's position on the board."""
        king_char = 'K' if player == 'w' else 'k'
        
        for row in range(8):
            for col in range(8):
                piece = board.get((row, col))
                if piece is not None and piece.piece == king_char:
                    return (row, col)
        
        return None
    
    def evaluate_move(self, move: Tuple[Tuple[int, int], Tuple[int, int], str]) -> int:
        """
        Evaluate a move based on material gain.
        """
        fr, fc, piece_type = move
        tr, tc = move[1]
        
        piece_value = self.PIECE_VALUES.get(piece_type, 0)
        
        # Check if capturing a piece
        target_row, target_col = tr, tc
        target_piece = self.board.get((target_row, target_col))
        
        if target_piece is not None:
            captured_value = self.PIECE_VALUES.get(target_piece.type, 0)
            piece_value += captured_value
        
        # Positional bonus for center
        if piece_type == PieceType.PAWN:
            # Calculate which rank the pawn is on
            if self.player == 'w':
                # White ranks are 1-8 in array (6-7 are ranks 2-1)
                piece_rank = (7 - fr) + 1
                center_bonus = 0
                if piece_rank == 5:
                    center_bonus = 30
                elif piece_rank == 4:
                    center_bonus = 50
            else:
                # Black
                piece_rank = fr + 1
                center_bonus = 0
                if piece_rank == 5:
                    center_bonus = 30
                elif piece_rank == 4:
                    center_bonus = 50
        
        return piece_value
    
    def get_best_move(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """Get the best legal move."""
        moves = self.get_all_legal_moves()
        
        if not moves:
            return None
        
        # Evaluate all moves and return the best one
        best_move = None
        best_score = float('-inf') if self.player == 'b' else float('inf')
        
        for move in moves:
            score = self.evaluate_move(move)
            if self.player == 'w':
                if score < best_score:
                    best_score = score
                    best_move = move
            else:
                if score > best_score:
                    best_score = score
                    best_move = move
        
        return best_move
    
    def apply_move(self, from_sq: Tuple[int, int], to_sq: Tuple[int, int]) -> Board:
        """Apply a move to create a new board."""
        piece = self.board.get(from_sq)
        if piece is None:
            return self.board
        
        new_board = Board()
        
        # Copy all pieces from old board
        for r in range(8):
            for c in range(8):
                p = self.board.get((r, c))
                if p is not None:
                    new_board.set((r, c), p)
        
        # Remove piece from from_sq
        new_board.set(from_sq, None)
        
        # Place piece on to_sq
        new_board.set(to_sq, piece)
        
        return new_board


def in_bounds(r: int, c: int) -> bool:
    """Check if position is on board."""
    return 0 <= r < 8 and 0 <= c < 8


def is_path_clear(from_sq: Tuple[int, int], to_sq: Tuple[int, int], board: Board) -> bool:
    """Check if path between squares is clear (for sliding pieces)."""
    fr, fc = from_sq
    tr, tc = to_sq
    
    if fr == tr:
        # Horizontal move
        c = min(fc, tc) + 1
        while c != tc:
            if board.get((fr, c)) is not None:
                return False
            c += 1
        return True
    else:
        # Vertical move
        r = min(fr, tr) + 1
        while r != tr:
            if board.get((r, fc)) is not None:
                return False
            r += 1
        return True


def is_path_clear_diagonal(from_sq: Tuple[int, int], to_sq: Tuple[int, int], board: Board) -> bool:
    """Check if diagonal path is clear."""
    fr, fc = from_sq
    tr, tc = to_sq
    
    dr = 1 if tr > fr else -1
    dc = 1 if tc > fc else -1
    
    r, c = fr + dr, fc + dc
    
    while (r, c) != (tr, tc):
        if board.get((r, c)) is not None:
            return False
        r += dr
        c += dc
    
    return True


# =============================================================================
# Main Game Loop
# =============================================================================


def play_game():
    """
    Play a chess game against the ChessEngine.
    """
    print("=" * 50)
    print("           CHESS ENGINE - PLAY AGAINST AI")
    print("=" * 50)
    
    print("\nChoose who plays first:")
    print("  1. You play as WHITE vs AI (Black)")
    print("  2. You play as BLACK vs AI (White)")
    print("  3. Quit")
    print("-" * 50)
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ['q', 'quit']:
        print("\nGoodbye! 👋")
        return
    
    player_color = 'w' if choice == '1' else 'b'
    ai_color = 'b' if player_color == 'w' else 'w'
    
    print(f"\nYou are playing as {player_color.upper()}")
    print(f"AI is playing as {ai_color.upper()}")
    
    # Initialize game
    game = ChessGame()
    
    # Set AI color
    game.turn = ai_color
    
    print("\n" + "=" * 50)
    print("                GAME START!")
    print("=" * 50)
    
    print_board(game.get_board())
    
    engine = ChessEngine(game.board, ai_color)
    
    move_num = 1
    max_moves = 100
    
    while move_num <= max_moves:
        current_player = game.get_turn()
        
        print(f"\n--- Move {move_num} --- ({current_player.upper()}) ---")
        print_board(game.get_board())
        
        if current_player == player_color:
            # Player's turn
            user_move = input("Your move: ").strip()
            
            if user_move.lower() in ['q', 'quit']:
                print("\nQuitting...")
                break
            
            # Validate move
            from_sq, to_sq = notation_to_coords(user_move[:2]), notation_to_coords(user_move[2:4])
            
            # Make player's move
            try:
                player_move = game.make_move(user_move)
                move_num += 1
            except ValueError as e:
                print(f"Invalid move: {e}")
                print(f"Please try again: {user_move}")
                continue
        
        else:
            # AI's turn
            best_move = engine.get_best_move()
            
            if best_move is None:
                print("AI has no legal moves!")
                break
            
            from_sq, to_sq = best_move[0], best_move[1]
            piece = game.board.get(from_sq)
            
            # Make AI's move
            try:
                game.make_move(to_sq)
                move_num += 1
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # Check game status
        if game.is_checkmate():
            winner = 'BLACK' if game.get_winner() == 'w' else 'WHITE'
            print("\n" + "=" * 50)
            print(f"CHECKMATE! {winner} WINS! 🏆")
            print("=" * 50)
            return
        
        if game.is_stalemate():
            print("\n" + "=" * 50)
            print("STALEMATE! It's a DRAW! 🤝")
            print("=" * 50)
            return
        
        if game.is_check():
            print("⚠️  CHECK!")
    
    if move_num >= max_moves:
        print("\n" + "=" * 50)
        print("DRAW by repetition (100 moves)!")
        print("=" * 50)
    
    print("\n" + "=" * 50)
    print_board(game.get_board())
    
    print("\nGame over! Goodbye! 👋")


if __name__ == '__main__':
    try:
        play_game()
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
