# Chess Engine in Python

A simple, type-safe chess engine implemented in Python.
Designed for clarity and educational purposes.

## Quick Start

```python
from chess import ChessGame, print_board, Piece, Board

# Create a new game (standard starting position)
game = ChessGame()

# View the board
print_board(game.get_board())

# Make moves
game.make_move("e4")  # White's opening move
game.make_move("e5")  # Black's response
print_board(game.get_board())
```

## Installation

```bash
cd chess
pip install -e .
```

## API

### `ChessGame` - The Main Game Engine

The `ChessGame` class manages the entire chess game state and handles:

- Turn management (White moves, then Black moves)
- Move parsing and validation
- Move execution
- Check, checkmate, and stalemate detection
- Special moves (castling, en passant, pawn promotion)

```python
from chess import ChessGame

# Initialize a new game
game = ChessGame()

# Make a move
move = game.make_move("e4")

# Check game status
game.is_check()          # Is the current player in check?
game.is_checkmate()      # Has the game ended in checkmate?
game.is_stalemate()      # Has the game ended in stalemate?
game.is_draw()           # Has the game ended in a draw?
game.get_winner()        # Returns 'w', 'b', or None
game.get_current_player()# Whose turn is it?
```

### `Board` - The Chess Board

```python
from chess import Board

# Create a new empty board
board = Board()

# Or create from standard starting position
board = Board.from_notation()

# Get piece at a square
piece = board.get((2, 4))  # e4 (row 2, col 4)

# Set piece at a square
board.set((2, 4), piece)   # Place piece on e4

# Convert algebraic notation to coordinates
row, col = board.to_coords("e4")
```

### `Piece` - Chess Pieces

```python
from chess import Piece, PieceType

# Create a white pawn
pawn = Piece(PieceType.PAWN, 'w')

# Create a black queen
queen = Piece(PieceType.QUEEN, 'b')

# Check piece color
pawn.is_white()  # True
pawn.is_black() # False

# Piece string representation
str(queen)  # 'Q' (uppercase for display)
str(pawn)   # 'P' (uppercase for display, but .type is 'p')
```

### `Move` - Move Objects

```python
from chess import Move, PieceType

# Create a move
move = Move(
    piece=Piece(PieceType.PAWN, 'w'),
    from_square=(2, 4),  # e4
    to_square=(4, 4),     # e5
    captured_piece=None,
    special='none'        # 'none', 'castling', 'en_passant', 'promotion'
)

# Check if move is legal
move.is_legal(board)  # Returns True or False

# Execute the move and get a new board state
new_board = move.execute(board)
```

### `PieceType` - The 6 Chess Piece Types

```python
from chess import PieceType

PieceType.PAWN  # 'p' - moves forward 1 (or 2 on first move), captures diagonally
PieceType.ROOK # 'r' - moves horizontally/vertically any distance
PieceType.KNIGHT# 'n' - moves in L-shape (2+1), can jump over pieces
PieceType.BISHOP# 'b' - moves diagonally any distance
PieceType.QUEEN # 'q' - combines Rook and Bishop movement (most powerful)
PieceType.KING  # 'k' - moves 1 square in any direction (most critical)
```

### `GameStatus` - Possible Game States

```python
from chess import GameStatus

GameStatus.ACTIVE     # Game is ongoing
GameStatus.CHECK      # Current player is in check
GameStatus.CHECKMATE  # Current player's king is in check and has no legal moves
GameStatus.STALEMATE  # Current player has no legal moves but is NOT in check
GameStatus.DRAW       # Game ended in a draw
GameStatus.WHITE_WON  # White won (checkmate)
GameStatus.BLACK_WON  # Black won (checkmate)
```

### `PlayerColor` - Player Colors

```python
from chess import PlayerColor

PlayerColor.WHITE  # 'w'
PlayerColor.BLACK  # 'b'

# Get the next player to move
PlayerColor.WHITE.next()    # Returns BLACK
PlayerColor.BLACK.next()    # Returns WHITE

# Get the opposite player
PlayerColor.WHITE.opposite()# Returns BLACK
```

## Chess Concepts Explained

### 1. Board Representation

The board is an 8x8 grid represented as a 2D array:

```python
board[row][col]
```

Where:

- `row 0` = Rank 8 (top of board)
- `row 7` = Rank 1 (bottom of board)
- `col 0` = File 'a' (left)
- `col 7` = File 'h' (right)

**Conversion to algebraic notation:**

```python
from_squares = board.to_coords("e4")  # Returns (3, 4)
# row = 7 - rank_number  (because array is 0-indexed, board is 1-indexed)
# col = file_letter_index (a=0, b=1, ..., h=7)
```

### 2. Piece Movement Rules

Each piece type has specific movement rules:

| Piece | Movement | Capture | Special |
| ------- | ---------- | --------- | --------- |
| Pawn | Forward 1 (2 on first move) | Diagonal | En passant, promotion |
| Rook | Straight lines (any distance) | Straight | Castling (with King) |
| Knight | L-shape (2+1 squares) | Any | None |
| Bishop | Diagonals (any distance) | Diagonal | None |
| Queen | Any direction (Rook + Bishop) | Any | None |
| King | 1 square in any direction | Any | Castling |

### 3. Check Detection

```python
# A player is in check if their king is under attack
is_in_check(board, PlayerColor.WHITE)
```

### 4. Checkmate

Checkmate is when:

1. The player's king is in check
2. The player has NO legal moves

```python
if game.is_checkmate():
    print("Checkmate! The game is over!")
    print(f"Winner: {game.get_winner()}")
```

### 5. Stalemate

Stalemate is when:

1. The player has NO legal moves
2. The player is NOT in check

```python
if game.is_stalemate():
    print("Stalemate! It's a draw.")
```

### 6. Castling (O-O and O-O-O)

Castling is a special move where the King moves 2 squares towards a Rook.

**Kingside castling (O-O):**

- King moves from e1 to g1
- Rook on h1 moves to f1
- King cannot be in check, must not pass through check, and must not end up in check
- King must not have moved before
- Rook on h-file must not have moved before
- h1 square must not be under attack
- f1 and g1 squares must not be under attack

**Queenside castling (O-O-O):**

- King moves from e1 to c1
- Rook on a1 moves to d1
- Similar safety requirements as kingside castling

**Implementation:**

```python
from chess import ChessGame

# White's turn
game.make_move("O-O")  # Kingside castling
# or
game.make_move("O-O-O") # Queenside castling

# Black's turn
game.make_move("o-o")  # Black's kingside (lowercase)
game.make_move("o-o-o")# Black's queenside
```

### 7. En Passant

En passant is a special pawn capture that can only happen immediately after:

- An opponent's pawn moves 2 squares forward from its starting position
- The pawn passes through the capture square diagonally

**Example:**

```text
Initial position:
P  # White pawn on e3
.  # Empty e4, e5
P  # Black pawn on e7

White plays: e4-e5
Black responds: e6xe6  # (en passant capture of white pawn)
```

**Requirements:**

- Must be white's move (since white's pawn moved 2 squares)
- Opponent's pawn must be on the en passant square
- Pawn must capture diagonally to where opponent's pawn was

### 8. Pawn Promotion

When a pawn reaches the opposite end of the board, it must be promoted to:

- Queen (most common)
- Rook
- Bishop
- Knight

```python
game.make_move("e7e8=Q")  # Promote pawn to Queen
game.make_move("e7e8=R")  # Promote pawn to Rook
game.make_move("e7e8=B")  # Promote pawn to Bishop
game.make_move("e7e8=N")  # Promote pawn to Knight
```

### 9. Move Notation

The engine supports standard algebraic notation:

| Format | Example | Meaning |
| -------- | -------- | --------- |
| Simple | e4 | Pawn moves from e2 to e4 |
| Capture | e5xe5 | (rare) |
| Capture | Nxe5 | Knight captures on e5 |
| Capture | e5xg7 | Pawn captures king |
| Disambiguated | Nge7 | Knight from g-file to e7 |
| Disambiguated | Nd2e7 | Knight from d2 to e7 |
| Castling | O-O | White kingside |
| Castling | O-O-O | White queenside |
| Castling | o-o | Black kingside |
| Castling | o-o-o | Black queenside |
| Promotion | e7e8=Q | Pawn promotes to Queen |

**Disambiguation:** When two or more pieces of the same type can move to the
same square, you must specify which one:

- By file: Nce7 vs Nge7
- By rank: Nd7e7 vs Nb7e7
- Both: Nc7d5 vs Nb7d5

### 10. Turn Management

The game alternates between players:

1. White moves
2. Black moves
3. White moves
4. ... and so on

**Checking whose turn it is:**

```python
if game.get_current_player() == PlayerColor.WHITE:
    print("White's turn!")
else:
    print("Black's turn!")
```

### 11. Captures

When you capture a piece, its piece is removed from the board.

```python
# After capturing, the opponent's piece is gone
game.make_move("Bxh7")  # Bishop captures knight on h7
# Black's knight is now removed from the board
```

## Usage Examples

### Play a Simple Game

```python
from chess import ChessGame, print_board

# Create game
game = ChessGame()

# Print initial board
print("=== Initial Position ===")
print_board(game.get_board())

# Open with Italian Game
print("\n=== Italian Game ===")
print("1. e4 e5")
game.make_move("e4")
game.make_move("e5")
print_board(game.get_board())

# 2. Nf3 Nc6
print("\n2. Nf3 Nc6")
game.make_move("Nf3")
game.make_move("Nc6")
print_board(game.get_board())

# 3. Bc4 Bc5
print("\n3. Bc4 Bc5")
game.make_move("Bc4")
game.make_move("Bc5")
print_board(game.get_board())

# 4. c3 Nge7
print("\n4. c3 Nge7")
game.make_move("c3")
game.make_move("Nge7")
print_board(game.get_board())
```

### Play Until Checkmate

```python
from chess import ChessGame

game = ChessGame()

# Play 50 moves
for i in range(50):
    if game.is_checkmate():
        break
    
    if game.is_my_turn('w'):
        move = get_my_move()  # Replace with actual logic
    else:
        move = get_opponent_move()  # Replace with actual logic
    
    game.make_move(move)
    
    if game.is_checkmate():
        break
    
    print(f"{i+1}. {move}")

# Check result
if game.is_checkmate():
    print(f"Checkmate! Winner: {game.get_winner()}")
```

### Check Game Status After Any Move

```python
game.make_move("e4")
if game.is_check():
    print("White is in check! Black must respond.")

game.make_move("e5")
if game.is_check():
    print("Black is in check! White must respond.")

if game.is_checkmate():
    print(f"Checkmate! {game.get_winner()} wins!")
```

## Strict Typing

The engine uses Python's type system heavily:

```python
from typing import Literal, List, Tuple, Union

Color = Literal['w', 'b']  # Only 'w' or 'b' allowed
Square = Tuple[int, int]  # Always (row, col)

# Type-safe piece creation
pawn: Piece = Piece(PieceType.PAWN, Color)  # Type checked
```

## Type Definitions

```python
# Colors
Color = Literal['w', 'b']  # White = 'w', Black = 'b'
PlayerColor = Union['w', 'b']

# Piece types
PieceType = Literal['p', 'r', 'n', 'b', 'q', 'k']

# Squares
Square = Tuple[int, int]  # (row, col), both 0-7

# Special move types
SpecialMove = Literal['none', 'castling', 'en_passant', 'promotion']

# Game status
GameStatus = Literal['active', 'check', 'checkmate', 'stalemate', 'draw']
```

## Data Structures

```python
class Piece:
    """Immutable chess piece with type and color."""
    type: PieceType
    color: Color
    # Pieces are immutable - type and color never change

class Board:
    """8x8 chess board represented as 2D array."""
    squares: List[List[Piece | None]]  # board[row][col]
    # board[row][col] = Piece or None

class Move:
    """Represents a move from one square to another."""
    piece: Piece
    from_square: Square
    to_square: Square
    captured_piece: Piece | None = None
    special: SpecialMove = 'none'
    castling: str | None = None  # 'k' or 'q' for castling
    promotion_to: PieceType | None = None

class GameState:
    """Complete game state."""
    board: Board
    current_player: Color
    castling_rights: Dict[str, Set[str]]  # {'w': {'kingside', 'queenside'}, ...}
    en_passant_target: Square | None
    status: GameStatus
```

## Special Moves

### Castling

```python
def castling_moves(player: Color, board: Board) -> List[Move]:
    """
    Generate all possible castling moves for a player.
    
    Args:
        player: 'w' for white, 'b' for black
        board: Current board state
    
    Returns:
        List of Move objects for castling
    """
    # Check if king can castle kingside or queenside
    # Verify:
    # - King hasn't moved
    # - Rook hasn't moved (for its file)
    # - King not in check
    # - King path safe (no attacks)
    # - King doesn't pass through check
    pass
```

### En Passant

```python
# After white plays e5, black can capture en passant:
game.set_en_passant((6, 4))  # Set en passant target square

# Black captures: e6xe6
move = Move(
    piece=Piece(PieceType.PAWN, 'b'),
    from_square=(5, 4),  # e7
    to_square=(4, 4),     # e6
    special='en_passant'
)
```

### Pawn Promotion

```python
# Pawn reaches rank 8 (row 0) or rank 1 (row 7)
game.make_move("e7e8=Q")  # Promote to Queen
game.make_move("e7e8=R")  # Promote to Rook
```

## Running the Engine

```python
from chess import ChessGame, print_board, PlayerColor

def play_game():
    game = ChessGame()
    
    while not game.is_checkmate() and not game.is_stalemate():
        # Your move logic here
        move = input(f"{game.get_current_player()}'s move: ")
        
        try:
            game.make_move(move)
            print_board(game.get_board())
        except ValueError as e:
            print(f"Invalid move: {e}")
            continue
        
        if game.is_check():
            print("Check!")
        
        if game.is_checkmate():
            print(f"Checkmate! {game.get_winner()} wins!")
            break
    
    print(f"\nFinal result: {game.game_state.status}")
    return game

# Play the game
play_game()
```

## License

MIT License - Feel free to use and modify!

## Contributing

Contributions are welcome! Please follow the coding style in the engine.

## Acknowledgments

- [FIDE Laws of Chess](https://www.fide.org/resources/laws-of-chess)
- [Chess.com - Chess Fundamentals](https://www.chess.com/articles/learn/game/rules)
- [Lichess Chess Programming Wiki](https://github.com/nicholaslane/chess.py)

---

Happy Chessing! ♟️
