#pragma once

#include "sentiment/core/types.hpp"
#include "sentiment/text/tokenizer.hpp"
#include "sentiment/text/vocabulary.hpp"

namespace sentiment {

class TfidfVectorizer {
public:
    explicit TfidfVectorizer(
        sz max_features = 50'000
    );

    void fit(
        const vec<str>& documents
    );

    [[nodiscard]]
    vec<double> transform(
        sv document
    ) const;

    [[nodiscard]]
    sz vocabulary_size() const noexcept;

private:
    Tokenizer tokenizer_;
    Vocabulary vocabulary_;

    sz max_features_;

    vec<double> idf_;

    bool fitted_{false};
};

} // namespace sentiment
