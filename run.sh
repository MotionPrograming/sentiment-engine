#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-$ROOT/data/software_reviews_3m.parquet}"
MODEL_DIR="${2:-$ROOT/models}"

if [[ ! -f "$DATASET" ]]; then
  echo "ERROR: dataset not found: $DATASET"
  echo "Put your Parquet dataset at: $ROOT/data/software_reviews_3m.parquet"
  exit 2
fi

cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build" -j"$(nproc)"
"$ROOT/build/sentiment_train" "$DATASET" "$MODEL_DIR"
