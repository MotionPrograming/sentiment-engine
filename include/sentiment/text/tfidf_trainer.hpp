#pragma once

#include "sentiment/core/types.hpp"
#include "sentiment/io/dataset_reader.hpp"
#include "sentiment/text/tokenizer.hpp"
#include "sentiment/text/tfidf_vectorizer.hpp"

namespace sentiment {

class TfidfTrainer {
public:
    explicit TfidfTrainer(
        sz max_features = 50'000
    );

    bool fit(
        ParquetDatasetReader& reader
    );

    [[nodiscard]]
    TfidfVectorizer finalize() &&;

    [[nodiscard]]
    sz vocabulary_size() const noexcept;

private:
    Tokenizer tokenizer_;

    sz max_features_;

    vec<u32> document_frequency_;

    // Temporary vocabulary used during DF collection.
    std::unordered_map<str, u32> df_;

    sz document_count_{0};
};

} // namespace sentiment