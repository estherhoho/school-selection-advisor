#!/bin/bash
# School Selection Advisor — One-time setup
# Run this once: bash setup.sh

set -e

echo "=== Setting up School Selection Advisor ==="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required. Install from https://www.python.org/downloads/"
    exit 1
fi

echo "Found Python: $(python3 --version)"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate and install dependencies
echo "Installing dependencies..."
source .venv/bin/activate
pip install -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo "To run the app:"
echo "  bash run.sh"
