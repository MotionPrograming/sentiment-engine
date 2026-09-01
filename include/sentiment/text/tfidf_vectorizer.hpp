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
    SparseVector transform_sparse(
        sv document
    ) const;

    [[nodiscard]]
    vec<double> transform(
        sv document
    ) const;

    [[nodiscard]]
    const Vocabulary&
    vocabulary() const noexcept;

    [[nodiscard]]
    const vec<double>&
    idf() const noexcept;

    [[nodiscard]]
    sz vocabulary_size() const noexcept;

    [[nodiscard]]
    bool fitted() const noexcept {
        return fitted_;
    }

    bool set_model_data(
        Vocabulary vocabulary,
        vec<double> idf
    );

    [[nodiscard]]
    bool save(
        const str& path
    ) const;

    [[nodiscard]]
    bool load(
        const str& path
    );

private:

    /*
     * Generate the same unigram + bigram representation
     * during fitting and inference.
     */
    void collect_term_ids(
        sv document,
        vec<u32>& ids
    ) const;

    Tokenizer tokenizer_;

    sz max_features_{50'000};

    Vocabulary vocabulary_;

    vec<double> idf_;

    bool fitted_{false};
};

} // namespace sentiment