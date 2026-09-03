#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
    echo 'Usage: ./scripts/predict.sh "review text"' >&2
    exit 2
fi

./build/sentiment_predict models "$1"
