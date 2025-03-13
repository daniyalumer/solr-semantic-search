#!/bin/bash
set -e  # Exit on error

# Download the dataset
pipenv run python3 data/download.py

# Get the cache directory path from the python script output
CACHE_PATH=$(pipenv run python3 data/download.py | grep "Path to dataset files:" | cut -d' ' -f5)

if [ -z "$CACHE_PATH" ]; then
    echo "Error: Could not get cache path from download script"
    exit 1
fi

# Check if source directory exists
if [ ! -d "$CACHE_PATH" ]; then
    echo "Error: Source directory not found: $CACHE_PATH"
    exit 1
fi

# Create target directory if it doesn't exist
mkdir -p data/resumepdf/data

# Use find to move files (handles spaces and special characters better)
find "$CACHE_PATH" -mindepth 1 -maxdepth 1 -exec mv {} data/resumepdf/data/ \;

echo "Dataset successfully moved to data/resumepdf/data/"