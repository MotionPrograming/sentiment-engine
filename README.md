```markdown
# SentimentEngine — Optimized C++ Sentiment Analysis

Production-oriented C++20 sentiment classifier featuring:

- **Unigram + Bigram** n-grams
- **TF-IDF** with smoothed IDF
- **Sublinear TF**: `1 + log(tf)`
- **L2 normalization**
- **Linear multiclass SVM (Crammer-Singer style)**
- **Neutral-class weighting**
- Deterministic Parquet dataset split + binary disk cache
- Reusable sparse feature buffers in hot loops
- Per-epoch shuffling for better SGD convergence
- Early stopping based on validation Macro F1 score
- Binary model persistence for ultra-fast load times

---

## Dataset

Place your Parquet dataset file inside the `data/` directory:

```text
data/software_reviews_3m.parquet

```

### Required Columns

| Column Name | Type |
| --- | --- |
| `review_text` | `string` |
| `sentiment` | `string` or `uint8` |

### Supported String Labels

* `negative`
* `neutral`
* `positive`

---

## Build

### Dependencies (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y build-essential cmake libarrow-dev libparquet-dev

```

### Compilation

Build the project using the build script:

```bash
./scripts/build.sh

```

Or configure manually via CMake:

```bash
cmake -B build -S .
cmake --build build -j$(nproc)

```

---

## Train

Run training using the helper script:

```bash
./scripts/train.sh data/software_reviews_3m.parquet

```

Or directly via the binary:

```bash
./build/sentiment_train data/software_reviews_3m.parquet models/ cache/

```

### Generated Artifacts

| Directory | Artifact | Description |
| --- | --- | --- |
| `cache/` | `train.bin` | Binary cached training dataset split |
| `cache/` | `validation.bin` | Binary cached validation dataset split |
| `cache/` | `test.bin` | Binary cached test dataset split |
| `cache/` | `manifest.bin` | Cache manifest & dataset checksum |
| `models/` | `tfidf.model` | Fitted vocabulary and IDF vector binary |
| `models/` | `svm.model` | Serialized Linear SVM weights & biases |

> **Note:** Cache is automatically invalidated when the source dataset changes or the cache version changes. TF-IDF vocabulary and IDF values are fitted exclusively on the training split to prevent data leakage.

---

## Predict

Run inference on sample text using the prediction script:

```bash
./scripts/predict.sh "This product is excellent and I love it"

```

Or run the binary directly:

```bash
./build/sentiment_predict models/ "This product is excellent and I love it"

```

The predictor outputs:

* Predicted sentiment class (`Negative`, `Neutral`, `Positive`)
* Multiclass decision scores
* Relative normalized confidence score
* Margin distance
* Action policy recommendation (`ACCEPT`, `REVIEW`, or `ABSTAIN`)

---

## Hyperparameters & Settings

### TF-IDF Vectorizer

* **Max Features**: `500,000`
* **Minimum Document Frequency (`min_df`)**: `3`
* **N-gram Range**: `1..2` (Unigrams + Bigrams)
* **Sublinear TF**: `true`
* **Normalization**: `L2`

### Training Config (`apps/train.cpp`)

* **Max Epochs**: `500`
* **Initial Learning Rate**: `0.0010`
* **Minimum Learning Rate**: `0.00008`
* **LR Decay Rate**: `0.12`
* **Regularization ($\lambda$)**: `0.00005`
* **Class Weights**: Negative: `1.0`, Neutral: `2.4`, Positive: `1.0`
* **Validation Interval**: `2` epochs
* **Early Stopping Patience**: `3` validations

---

## Performance Optimizations

* **Allocation-Free Hot Loops**: The training and validation passes reuse caller-owned `SparseVector` buffers to eliminate millions of dynamic memory allocations during epoch passes.
* **Header-Level Inlining**: Optimized `const std::string&` and `std::string_view` overloads in `Vocabulary` avoid unnecessary string allocations during token lookups.
* **Deterministic Shuffling**: Per-epoch sample shuffling using fixed-seed RNG improves SGD convergence stability.

---

## Testing & Maintenance

### Run Test Suite

```bash
ctest --test-dir build --output-on-failure

```

### Clean Rebuild

```bash
rm -rf build
./scripts/build.sh

```

### Clear Dataset Cache

```bash
rm -f cache/*.bin
./scripts/train.sh data/software_reviews_3m.parquet

```

---

## Repository Structure

```text
sentiment-engine/
├── apps/            # Application entrypoints (train.cpp, predict.cpp)
├── benchmarks/      # Performance benchmark scripts & binaries
├── cache/           # Serialized binary dataset splits
├── data/            # Parquet dataset storage
├── include/         # Header files (sentiment/core, sentiment/ml, sentiment/text)
├── models/          # Exported TF-IDF and SVM model binaries
├── scripts/         # Helper shell scripts (build.sh, run.sh, predict.sh)
├── src/             # Implementation C++ source files
├── CMakeLists.txt   # CMake build configuration
└── README.md        # Project documentation

```

```

<ElicitationsGroup message="What would you like to do next?">
  <Elicitation label="Update the README file directly in your workspace" query="Overwrite README.md with the updated content"/>
  <Elicitation label="Run training on software_reviews_3m.parquet" query="Execute sentiment_train with data/software_reviews_3m.parquet"/>
</ElicitationsGroup>

```
