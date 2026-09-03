# SentimentEngine — Optimized C++ Sentiment Analysis

Production-oriented C++20 sentiment classifier using:

- **Unigram + Bigram** n-grams
- **TF-IDF** with smoothed IDF
- **Sublinear TF**: `1 + log(tf)`
- **L2 normalization**
- **Linear multiclass SVM (Crammer-Singer style)**
- Neutral-class weighting
- Deterministic Parquet dataset split + disk cache
- Reusable sparse feature buffers in hot loops
- Per-epoch shuffling for better SGD convergence
- Early stopping on validation Macro F1
- Binary model persistence

## Dataset

Put your Parquet file at:

```text
data/reviews.parquet
```

Required columns:

```text
review_text : string
sentiment   : string or uint8
```

String labels:

```text
negative
neutral
positive
```

## Build

Ubuntu/Linux Mint:

```bash
sudo apt update
sudo apt install -y build-essential cmake libarrow-dev libparquet-dev
```

Then:

```bash
./scripts/build.sh
```

If Arrow/Parquet are installed in a custom prefix, pass the appropriate `CMAKE_PREFIX_PATH` to CMake.

## Train

```bash
./scripts/train.sh data/reviews.parquet
```

Or directly:

```bash
./build/sentiment_train data/reviews.parquet models cache
```

Generated artifacts:

```text
cache/train.bin
cache/validation.bin
cache/test.bin
cache/manifest.bin
models/tfidf.model
models/svm.model
```

The cache is automatically invalidated when the dataset changes or the cache version changes.

**Important:** TF-IDF vocabulary and IDF are learned from the training split only. Validation/test rows never influence the vocabulary.

## Predict

```bash
./scripts/predict.sh "This product is excellent and I love it"
```

Or:

```bash
./build/sentiment_predict models "This product is excellent and I love it"
```

The predictor reports sentiment, decision scores, relative confidence, margin, and a simple ACCEPT/REVIEW/ABSTAIN policy.

Confidence is a score-based indicator, **not a calibrated probability**.

## TF-IDF

Defaults:

```text
max_features = 50,000
min_df       = 3
ngram_range  = 1..2
sublinear_tf = true
normalization = L2
```

So both unigrams and adjacent bigrams are features.

## Training

Main settings are in `apps/train.cpp`:

```text
Epochs              = 8
Initial LR          = 0.0010
Minimum LR          = 0.00008
LR decay            = 0.12
Regularization      = 0.00005
Neutral weight      = 3.80
Validation interval = 2
Early stopping      = 3
```

Tune these against validation Macro F1 for your dataset.

## Optimization changes

The hot training/evaluation loops reuse a caller-owned `SparseVector`, avoiding repeated output-vector allocations.

Vocabulary lookup has a `const string&` overload, avoiding unnecessary string construction when the token is already a `string`.

Training order is shuffled once per epoch with a deterministic RNG seed.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

## Clean rebuild

```bash
rm -rf build
./scripts/build.sh
```

Force a fresh dataset cache:

```bash
rm -f cache/*.bin
./scripts/train.sh data/reviews.parquet
```

## Layout

```text
sentiment-engine/
├── apps/
├── include/
├── src/
├── tests/
├── data/        # put your dataset here
├── models/      # generated model files
├── cache/       # generated dataset cache
├── scripts/
├── CMakeLists.txt
└── README.md
```
