#!/usr/bin/env bash
set -e

echo "🔧 Setting up Sandman project environment..."

# 1. Ensure uv is installed
if ! command -v uv &> /dev/null; then
  echo "❌ uv is not installed."
  echo "Install it from https://docs.astral.sh/uv/"
  exit 1
fi

# 2. Create virtual environment if missing
if [ ! -d ".venv" ]; then
  echo "📦 Creating virtual environment..."
  uv venv
else
  echo "✅ Virtual environment already exists."
fi

# 3. Install package in editable mode
echo "📦 Installing sandman package (editable)..."
cd SANDMAN
uv pip install -e .

cd ..