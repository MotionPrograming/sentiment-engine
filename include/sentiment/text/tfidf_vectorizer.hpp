#pragma once

#include "sentiment/core/types.hpp"
#include "sentiment/text/tokenizer.hpp"
#include "sentiment/text/vocabulary.hpp"

#include <functional>

namespace sentiment {

class TfidfVectorizer {
public:
    explicit TfidfVectorizer(
        sz max_features = 50'000
    );

    void fit(
        const vec<str>& documents
    );

    void fit_stream(
        sz document_count,
        const std::function<bool(str&)>& next_document
    );

    [[nodiscard]]
    vec<double> transform(
        sv document
    ) const;

    [[nodiscard]]
    SparseVector transform_sparse(
        sv document
    ) const;

    [[nodiscard]]
    sz vocabulary_size() const noexcept;

    [[nodiscard]]
    const Vocabulary& vocabulary() const noexcept;

    [[nodiscard]]
    const vec<double>& idf() const noexcept;

    bool set_model_data(
        Vocabulary vocabulary,
        vec<double> idf
    );

private:
    Tokenizer tokenizer_;
    Vocabulary vocabulary_;

    sz max_features_;

    vec<double> idf_;

    bool fitted_{false};
};

} // namespace sentiment
