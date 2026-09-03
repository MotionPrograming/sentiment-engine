#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${1:-$ROOT/models}"
TEXT="${2:-This product is excellent and I really love it}"
"$ROOT/build/sentiment_predict" "$MODEL_DIR" "$TEXT"
