"""
Simple Chess Engine - A basic AI that plays legal chess moves.

This engine generates all legal moves for chess according to the rules:
- Pieces move according to their movement patterns
- Sliding pieces (R, B, Q) cannot jump
- King cannot move into or through check
- Castling rules (safe, king not in check)
- En passant (only valid on specific conditions)
- Pawn promotion (forced to move when reaching the last rank)

To play: python3 chess/simple_engine.py
"""

import random
from chess import ChessGame, print_board
from chess.move import Move as RealMove, file_letter, rank_letter, is_square_attacked


def can_castle(board, player, side):
    """
    Check if a player can castle on a given side.
    
    Castling is valid when:
    - The king is on its starting square
    - The king hasn't moved yet
    - The king is not in check
    - The king doesn't pass through or end in check
    - The rook on the castling side hasn't moved
    """
    piece_map = {
        'k': PieceType.KING, 'K': PieceType.KING,
        'q': PieceType.QUEEN, 'Q': PieceType.QUEEN,
        'r': PieceType.ROOK, 'R': PieceType.ROOK,
        'p': PieceType.PAWN, 'P': PieceType.PAWN,
        'n': PieceType.KNIGHT, 'N': PieceType.KNIGHT,
        'b': PieceType.BISHOP, 'B': PieceType.BISHOP,
    }
    
    if player == 'w':
        starting_sq = (0, 4)  # e1
        rook_kingside_start = (0, 7)  # h1
        rook_queenside_start = (0, 0)  # a1
    else:
        starting_sq = (7, 4)  # e8
        rook_kingside_start = (7, 7)  # h8
        rook_queenside_start = (7, 0)  # a8
    
    king = board.get(starting_sq)
    
    # King must be in starting position
    if king is None or king.type.name != 'KING':
        return False, None, None
    
    # King must not be in check
    if is_square_attacked(starting_sq, player, board):
        return False, None, None
    
    if side == 'kingside':
        rook = board.get(rook_kingside_start)
        # Kingside: rook must exist and king doesn't pass through/e8 (h7 for black, g6/g5/g4/g3 for white)
        if rook is None:
            return False, None, None
        
        # Check if king passes through check
        # White: e1 -> f1 (check h8 and h4-h3)
        # Black: e8 -> f8 (check h7 and h5-h4-h3)
        if player == 'w':
            if is_square_attacked((0, 5), player, board) or is_square_attacked((0, 6), player, board):
                return False, None, None
        else:
            if is_square_attacked((7, 5), player, board) or is_square_attacked((7, 6), player, board):
                return False, None, None
    
    elif side == 'queenside':
        rook = board.get(rook_queenside_start)
        # Queenside: rook must exist and king doesn't pass through/e8 (c7 for black, d6/d5/d4/d3 for white)
        if rook is None:
            return False, None, None
        
        # Check if king passes through check
        # White: e1 -> d1 -> c1 (check h8 and f3, e3, d3)
        # Black: e8 -> d8 -> c8 (check h7 and f6, e6, d6)
        if player == 'w':
            if is_square_attacked((0, 3), player, board) or is_square_attacked((0, 2), player, board):
                return False, None, None
        else:
            if is_square_attacked((7, 3), player, board) or is_square_attacked((7, 2), player, board):
                return False, None, None
    
    return True, starting_sq, side


def execute_castling(board, player, side):
    """Execute a castling move and return a new board with king and rook moved."""
    row = 0 if player == 'w' else 7
    side_map = {'kingside': 1, 'queenside': 3}
    
    new_board = Board()
    
    # Copy all pieces from old board
    for r in range(8):
        for c in range(8):
            piece = board.get((r, c))
            if piece is not None:
                new_board.set((r, c), piece)
    
    # Move king
    king_col = 4 + side_map[side]  # 6 for kingside (g), 2 for queenside (c)
    new_board.set((row, king_col), Piece(PieceType.KING, player))
    
    # Move rook
    rook_col = 3 + side_map[side]  # 5 for kingside (f), 3 for queenside (d)
    if side == 'queenside':
        new_board.set((row, side_map[side]), Piece(PieceType.ROOK, player))
    else:
        new_board.set((row, 7 - side_map[side]), Piece(PieceType.ROOK, player))
    
    return new_board


def evaluate_square(board, square, player):
    """
    Evaluate a square based on piece values and position.
    
    Returns a score for a piece on that square.
    """
    piece = board.get(square)
    if piece is None:
        return 0
    
    piece_values = {
        PieceType.PAWN: 100,
        PieceType.ROOK: 500,
        PieceType.KNIGHT: 320,
        PieceType.BISHOP: 330,
        PieceType.QUEEN: 900,
        PieceType.KING: 20000,
    }
    
    # Basic positional bonus
    # Center control
    row, col = square
    center_score = 0
    if row in [3, 4] and col in [3, 4]:
        center_score = 10
    
    # Pawn advantage (favor center pawns)
    if piece.type == PieceType.PAWN:
        pawn_center_bonus = {
            (0, 0): 0, (0, 1): 0, (0, 2): 15, (0, 3): 30, (0, 4): 40, (0, 5): 30, (0, 6): 15, (0, 7): 0,
            (1, 0): 0, (1, 1): 0, (1, 2): 15, (1, 3): 30, (1, 4): 40, (1, 5): 30, (1, 6): 15, (1, 7): 0,
            (2, 0): 0, (2, 1): 5,  (2, 2): 20, (2, 3): 35, (2, 4): 50, (2, 5): 35, (2, 6): 20, (2, 7): 5,
            (3, 0): 0, (3, 1): 5,  (3, 2): 20, (3, 3): 35, (3, 4): 50, (3, 5): 35, (3, 6): 20, (3, 7): 5,
        }
        center_score += pawn_center_bonus.get(square, 0)
    
    # Knight opening/closing phase bonus
    knight_phases = {
        (0, 1): 30, (0, 6): 30,
        (1, 2): 25, (1, 5): 25,
        (1, 3): 35, (1, 4): 35,
        (2, 1): 30, (2, 6): 30,
        (3, 2): 30, (3, 5): 30,
        (4, 3): 40, (4, 4): 40,
        (5, 2): 30, (5, 5): 30,
        (6, 1): 30, (6, 6): 30,
        (7, 2): 25, (7, 5): 25,
        (7, 1): 30, (7, 6): 30,
    }
    knight_bonus = knight_phases.get(square, 0)
    
    # Bishop opening/closing phase bonus
    bishop_phases = {
        (0, 3): 30, (0, 4): 35, (0, 5): 30,
        (1, 1): 20, (1, 6): 20,
        (2, 0): 0,  (2, 7): 0,
        (6, 1): 20, (6, 6): 20,
        (7, 3): 30, (7, 4): 35, (7, 5): 30,
    }
    bishop_bonus = bishop_phases.get(square, 0)
    
    # Center control bonus (squares near center)
    center_control = abs(row - 3.5) + abs(col - 3.5)
    
    return piece_values.get(piece.type, 0) + center_score + knight_bonus + bishop_bonus - center_control


def square_is_controlling(board, square, player):
    """
    Check if the player has pieces controlling this square.
    
    A player controls a square if:
    - They have a piece on that square, or
    - They have a piece that can move to that square
    """
    # Check for pieces on the square
    piece = board.get(square)
    if piece is not None and piece.color == player:
        return True
    
    # Check for pieces that can move to this square
    piece_map = {
        PieceType.PAWN: {'direction': 1 if player == 'w' else -1},
        PieceType.ROOK: {},
        PieceType.KNIGHT: {'moves': [(2, 1), (1, 2), (-1, 2), (-2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]},
        PieceType.BISHOP: {'directions': [[1, 1], [1, -1], [-1, 1], [-1, -1]]},
        PieceType.QUEEN: {'directions': [[1, 1], [1, -1], [-1, 1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]]},
        PieceType.KING: {'directions': [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]},
    }
    
    if piece is not None:
        piece_info = piece_map.get(piece.type)
        if piece_info:
            piece_type = piece.type.name
            if piece_type == 'PAWN':
                direction = piece_info['direction']
                # Check if we can move to this square from our piece
                if square[0] == piece[0] + 2 * direction or square[0] == piece[0] + direction:
                    return True
            elif piece_type == 'KNIGHT':
                knight_moves = piece_info['moves']
                for dr, dc in knight_moves:
                    if (square[0] == piece[0] + dr and square[1] == piece[1] + dc):
                        return True
            elif piece_type in ['ROOK', 'BISHOP', 'QUEEN']:
                for d_row, d_col in piece_info['directions']:
                    # Check if we can reach this square
                    # Skip the current piece square
                    row, col = square
                    pr, pc = piece
                    if d_row == 0 and d_col == 0:
                        continue
                    step_r, step_c = d_row, d_col
                    while True:
                        if (row == pr + r * step_r and col == pc + c * step_c):
                            return True
                        r += 1
                        c += 1
                    break
                return True
            elif piece_type == 'KING':
                king_directions = piece_info['directions']
                for dr, dc in king_directions:
                    if abs(square[0] - piece[0]) <= 1 and abs(square[1] - piece[1]) <= 1:
                        return True
                return True
    
    return False


class EngineMove:
    """
    Represents a move from our engine's perspective.
    
    Attributes:
        piece_type: The type of piece being moved
        from_file: Starting file ('a'-'h')
        to_file: Destination file ('a'-'h')
        from_rank: Starting rank (1-8)
        to_rank: Destination rank (1-8)
        capture: True if this move captures a piece
        castling_side: 'kingside' or 'queenside' if castling, else None
        promoted: True if pawn promoted, and what it promoted to
        en_passant: True if this is an en passant capture
    """
    def __init__(self, piece_type, from_file, to_file, from_rank, to_rank, 
                 capture=False, castling_side=None, promoted=False, promotion=None, en_passant=False):
        self.piece_type = piece_type
        self.from_file = from_file
        self.to_file = to_file
        self.from_rank = from_rank
        self.to_rank = to_rank
        self.capture = capture
        self.castling_side = castling_side
        self.promoted = promoted
        self.promotion = promotion
        self.en_passant = en_passant
    
    def get_score(self, board, player):
        """Calculate a score for this move."""
        score = 0
        
        if self.capture:
            # Capture score
            capture_values = {'p': 100, 'r': 500, 'n': 320, 'b': 330, 'q': 900, 'k': 10000}
            captured = board.get((self.to_rank - 1, self.to_file - 'a'.index(self.to_file)))
            if captured:
                score += capture_values.get(captured.type.name, 0)
        
        if self.castling_side:
            # Castling is a good move (safe king)
            score += 500
        
        if self.promoted:
            # Promotion is always good
            score += 500
        
        # Positional evaluation
        # Evaluate the board with and without this move
        board1 = board
        board2 = board.copy()
        
        # Apply the move temporarily
        from_sq = (self.from_rank - 1, self.from_file - 'a'.index(self.to_file))
        to_sq = (self.to_rank - 1, self.to_file - 'a'.index(self.to_file))
        row = self.from_rank - 1
        
        for r in range(8):
            for c in range(8):
                if r == from_sq[0] and c == from_sq[1]:
                    piece = board1.get((r, c))
                    if piece is not None:
                        score -= evaluate_square(board1, (r, c), player)
                elif r == to_sq[0] and c == to_sq[1]:
                    piece = board1.get((r, c))
                    if piece is not None:
                        score += evaluate_square(board1, (r, c), player)
        
        return score + 2000 if player == 'w' else score - 2000  # Bonus for material gain


# =============================================================================
# Simple AI Engine
# =============================================================================


def generate_moves(game, player):
    """
    Generate all legal moves for a player.
    """
    moves = []
    
    board = game.get_board()
    turn = game.get_turn()
    
    # Castling moves
    can_castle_w, castle_start_w, castle_side_w = can_castle(board, 'w', 'kingside')
    can_castle_b, castle_start_b, castle_side_b = can_castle(board, 'b', 'kingside')
    
    if turn == 'w' and can_castle_w:
        castle_sq = (0, 4)
        moves.append(RealMove(
            piece=Piece(PieceType.KING, 'w'),
            from_square=castle_sq,
            to_square=(0, 4 + ('kingside' if can_castle_w else 2)),
        ))
    else:
        castle_sq = (7, 4)
        if player == 'b' and can_castle_b:
            moves.append(RealMove(
                piece=Piece(PieceType.KING, 'b'),
                from_square=castle_sq,
                to_square=(7, 4 + ('kingside' if can_castle_b else 2)),
            ))
    
    # Queen side castling
    if turn == 'w' and can_castle_w:
        castle_sq = (0, 4)
        moves.append(RealMove(
            piece=Piece(PieceType.KING, 'w'),
            from_square=castle_sq,
            to_square=(0, 4 - 2),
        ))
    else:
        castle_sq = (7, 4)
        if player == 'b' and can_castle_b:
            moves.append(RealMove(
                piece=Piece(PieceType.KING, 'b'),
                from_square=castle_sq,
                to_square=(7, 4 - 2),
            ))
    
    # Check all squares for pieces
    for row in range(8):
        for col in range(8):
            piece = board.get((row, col))
            if piece is None:
                continue
            if piece.color != turn:
                continue
            
            # Generate possible moves for this piece
            possible_moves = generate_piece_moves(piece, (row, col), board)
            
            # Filter out moves that leave king in check
            for move in possible_moves:
                test_board = board.copy()
                test_board.set((row, col), None)
                test_board.set((move.to_sq[0], move.to_sq[1]), move.piece)
                
                if not is_square_attacked((row, col), turn, test_board):
                    if not is_square_attacked(move.to_sq, turn, test_board):
                        moves.append(move)
                        move.to_sq = (row, col)
    
    return moves


# =============================================================================
# Game Engine
# =============================================================================


class ChessGame:
    """A simple chess game engine with basic AI."""
    
    def __init__(self, board=None):
        self.board = board or Board.from_notation()
        self.turn = 'w'
        self.status = GameStatus.ACTIVE
    
    def make_move(self, move):
        """Make a move on the board."""
        self.board = self._apply_move(move)
        
        # Switch turn
        self.turn = 'b' if self.turn == 'w' else 'w'
        
        # Check for checkmate/stalemate
        if is_in_check(self.board, self.turn):
            self.status = GameStatus.CHECK
        elif len(self._get_legal_moves()) == 0:
            self.status = GameStatus.STALEMATE
        else:
            self.status = GameStatus.ACTIVE
        
        return move
    
    def _apply_move(self, move):
        """Apply a move and return a new board."""
        from_sq = move.from_square
        to_sq = move.to_square
        
        new_board = Board()
        
        # Copy all pieces
        for row in range(8):
            for col in range(8):
                piece = self.board.get((row, col))
                if piece is not None:
                    new_board.set((row, col), piece)
        
        # Handle en passant
        if move.special == 'en_passant':
            ep_sq = (move.ep_row, move.ep_col)
            new_board.set(ep_sq, None)
        
        # Handle castling
        if move.special == 'castling':
            if move.piece.color == 'w':
                # Move rook
                if move.to_sq[1] == 6:  # Kingside
                    new_board.set((0, 6), Piece(PieceType.KING, 'w'))
                    new_board.set((0, 5), Piece(PieceType.ROOK, 'w'))
                else:  # Queenside
                    new_board.set((0, 2), Piece(PieceType.KING, 'w'))
                    new_board.set((0, 3), Piece(PieceType.ROOK, 'w'))
            else:
                if move.to_sq[1] == 6:
                    new_board.set((7, 6), Piece(PieceType.KING, 'b'))
                    new_board.set((7, 5), Piece(PieceType.ROOK, 'b'))
                else:
                    new_board.set((7, 2), Piece(PieceType.KING, 'b'))
                    new_board.set((7, 3), Piece(PieceType.ROOK, 'b'))
        
        # Handle promotion
        if move.special == 'promotion':
            new_board.set((move.to_sq[0], move.to_sq[1]),
                         Piece(move.promotion, move.piece.color))
        
        return new_board
    
    def _get_legal_moves(self):
        """Get all legal moves."""
        return []  # Placeholder - full implementation needed
    
    def is_check(self):
        """Is the current player in check?"""
        return is_in_check(self.board, self.turn)
    
    def is_checkmate(self):
        """Is the game in checkmate?"""
        return self.status == GameStatus.CHECKMATE
    
    def is_stalemate(self):
        """Is the game in stalemate?"""
        return self.status == GameStatus.STALEMATE
    
    def get_winner(self):
        """Get the winner if any."""
        if self.status == GameStatus.CHECKMATE:
            return 'b' if self.turn == 'w' else 'w'
        return None


def main():
    """Play a game against the simple engine."""
    import sys
    
    print("=" * 50)
    print("         CHESS ENGINE - TWO PLAYER GAME")
    print("=" * 50)
    print("\nChoose who plays first:")
    print("  1. You play as White")
    print("  2. You play as Black")
    print("  3. Quit")
    print("-" * 50)
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice in ['q', 'quit']:
        print("\nGoodbye! 👋")
        return
    
    player_color = 'w' if choice == '1' else 'b'
    
    print("\n" + "=" * 50)
    print("           NEW GAME - YOU ARE", player_color.upper())
    print("=" * 50)
    
    game = ChessGame()
    
    print_board(game.get_board())
    print(f"\nIt's your turn as {player_color.upper()}! Make your move.")
    print("Type 'q' to quit at any time.\n")
    
    move_num = 1
    max_moves = 100
    
    while move_num <= max_moves:
        # Player's turn
        print(f"\n--- Move {move_num} --- ({player_color.upper()}) ---")
        
        user_input = input("Your move: ").strip().lower()
        
        if user_input == 'q':
            print("\nQuitting...")
            break
        
        try:
            move = RealMove(
                piece=Piece(PieceType.PAWN, player_color),
                from_square=(3, 4),  # e2 for white or e7 for black
                to_square=(4, 4),    # e4 or e5
                special='none'
            )
            game.make_move(move)
            move_num += 1
            if game.turn != player_color:
                player_color = 'b' if player_color == 'w' else 'w'
            print(f"Move {move_num}: {player_color.upper()} to {file_letter(move.to_sq[1])}{rank_letter(move.to_sq[0])}")
        except Exception as e:
            print(f"Error: {e}")
            print("Try again!")
            continue
        
        if game.is_checkmate():
            print("\n" + "=" * 50)
            print("CHECKMATE! You win! 🏆")
            print("=" * 50)
            break
        
        if game.is_stalemate():
            print("\n" + "=" * 50)
            print("STALEMATE! It's a draw! 🤝")
            print("=" * 50)
            break
        
        # Engine's turn
        print(f"\n--- Engine's move ---")
        print_board(game.get_board())
        
        # Get legal moves
        moves = generate_moves(game, 'b')
        if not moves:
            if game.is_check():
                print("\nEngine in check - no legal moves!")
                break
            else:
                print("\nEngine cannot make a move - stalemate!")
                break
        
        # Choose the best move
        best_move = moves[0]
        best_score = best_move.get_score(game.board, 'b')
        
        for move in moves:
            score = move.get_score(game.board, 'b')
            if score > best_score:
                best_score = score
                best_move = move
        
        game.make_move(best_move)
        move_num += 1
        player_color = 'w'
        
        if game.is_check():
            print("⚠️  Check!")
        
        print(f"\n--- Move {move_num} --- ({player_color.upper()}) ---\n")
    
    if move_num > max_moves:
        print("\n" + "=" * 50)
        print("DRAW by repetition! 🤝")
        print("=" * 50)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
