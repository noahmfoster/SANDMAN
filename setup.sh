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

# 4. Download datasets if passed --download-data flag
if [ "$1" == "--download-data" ]; then
    echo "Checking and downloading datasets..."
    mkdir -p data
    # Download Mice (IBL)
    if [ ! -d "data/mice-ibl" ]; then
        echo "The ibl dataset comes from running data_loader_ibl.py script which downloads data from IBL alyx server."
        # Confirm with user whether to proceed
        read -p "⬇️  Do you want to proceed with downloading the Mice (IBL) dataset? (y/n): " confirm
        if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
            echo "⬇️  Downloading Mice (IBL) dataset..."
            mkdir -p data/mice-ibl
            uv run python SANDMAN/src/sandman/data_loading/data_loader_ibl.py
        else
            echo "❌ Skipping Mice (IBL) dataset download."
        fi
    else
        echo "✅ Mice (IBL) dataset already exists."
    fi
    # Download Mice (MAP)
    if [ ! -d "data/mice-MAP" ]; then
        echo "⬇️  Downloading Mice (MAP) dataset..."
        mkdir -p "data/mice-MAP"
        uv run dandi download DANDI:000363/0.230822.0128 -o "data/mice-MAP"
    else
        echo "✅ Mice (MAP) dataset already exists."
    fi
    mkdir -p data/synthetic
else
    echo "ℹ️  Skipping dataset download. To download datasets, run setup.sh with --download-data flag."
fi  

echo "🎉 Setup complete!"