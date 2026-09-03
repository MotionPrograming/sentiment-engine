#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET="${1:-data/reviews.parquet}"
MODEL_DIR="${2:-models}"
CACHE_DIR="${3:-cache}"

if [[ ! -f "$DATASET" ]]; then
    echo "ERROR: Dataset not found: $DATASET" >&2
    echo "Put your Parquet dataset at data/reviews.parquet or pass its path." >&2
    exit 2
fi

mkdir -p "$MODEL_DIR" "$CACHE_DIR"
./build/sentiment_train "$DATASET" "$MODEL_DIR" "$CACHE_DIR"
