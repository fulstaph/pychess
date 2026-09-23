#!/bin/bash

# Chess Engine - Installation Script
# This script installs the chess engine using uv

echo "===================================="
echo "  Chess Engine Installation"
echo "===================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo ""
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Use uv to install
echo "Installing chess engine with uv..."
echo ""

uv sync

echo ""
echo "===================================="
echo "  Installation Complete!"
echo "===================================="
echo ""
echo "To play the game:"
echo ""
echo "  # Run the simple engine:"
echo "  python3 chess/simple_engine.py"
echo ""
echo "  # Play with your name:"
echo "  python3 chess/play_game.py"
echo ""
echo "  # Or play directly in the REPL:"
echo "  python3"
echo "  >>> from chess import ChessGame, print_board"
echo "  >>> game = ChessGame()"
echo "  >>> print_board(game.get_board())"
echo "  >>> game.make_move('e4')"
echo ""
echo "Happy chessing! ♟️ "
