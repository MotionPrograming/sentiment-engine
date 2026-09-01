# SentimentEngine — optimized C++ sentiment classifier

## Dataset

Put the dataset here:

```text
data/software_reviews_3m.parquet
```

The Parquet file must contain:

- `review_text` — UTF-8 string
- `sentiment` — either `negative` / `neutral` / `positive` string, or UInt8 `0/1/2`

## Run from a clean checkout

```bash
chmod +x run.sh predict.sh
./run.sh
```

Or provide the dataset path explicitly:

```bash
./run.sh /path/to/software_reviews_3m.parquet
```

The trained artifacts are written to `models/`.

## Predict after training

```bash
./predict.sh "This software is fantastic and works perfectly"
./predict.sh "This software is terrible and crashes constantly"
./predict.sh "It is okay, nothing special"
```

## What was fixed

- Streaming TF-IDF vocabulary construction; the complete training corpus is no longer stored as millions of `std::string`s just to calculate DF.
- Duplicate reviews are removed before split assignment.
- Duplicate detection uses a canonical token representation, so case/punctuation variants such as `Great!` and `great` are treated as the same review.
- Conflicting labels for the same canonical review are rejected from the dataset instead of leaking contradictory supervision.
- Split assignment is deterministic and hash-based 80/10/10, avoiding the old row-order split bias and guaranteeing that duplicates cannot land in different splits.
- Sparse TF-IDF remains sorted and duplicate feature IDs are aggregated once.
- Linear SVM uses sparse updates and STL containers.
- Validation checkpoint + early stopping is retained.
- Final test set is evaluated only after restoring the best validation checkpoint.
- Trained TF-IDF and SVM models are saved to disk.
- `predict.cpp` no longer uses a toy hard-coded training set; it loads the real trained model and predicts arbitrary text.
- Prediction output includes class, score, and normalized softmax-style confidence values.
- No `bits/stdc++.h` dependency in the main public interfaces.

## Architecture

```text
Parquet
  ↓
stream + canonical duplicate filter
  ↓
deterministic hash split 80/10/10
  ↓
TRAIN only → streaming DF → TF-IDF vocabulary
  ↓
sparse TF-IDF
  ↓
weighted multiclass Linear SVM
  ↓
validation checkpoint / early stopping
  ↓
final untouched test benchmark
  ↓
models/tfidf.model + models/svm.model
```
