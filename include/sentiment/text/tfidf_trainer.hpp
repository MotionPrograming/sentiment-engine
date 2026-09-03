#pragma once

#include "sentiment/core/types.hpp"
#include "sentiment/math/sparse_vector.hpp"
#include "sentiment/text/tokenizer.hpp"
#include "sentiment/text/vocabulary.hpp"
#include "sentiment/math/sparse_vector.hpp"

namespace sentiment {

class TfidfVectorizer {
public:

    // ========================================================
    // Constructor
    // ========================================================

    explicit TfidfVectorizer(
        sz max_features = 50'000,
        sz min_df = 3,
        sz ngram_min = 1,
        sz ngram_max = 2,
        bool sublinear_tf = true
    );

    // ========================================================
    // Fit & Transform
    // ========================================================

    void fit(
        const vec<str>& documents
    );

    [[nodiscard]]
    vec<double> transform(
        sv document
    ) const;

    [[nodiscard]]
    SparseVector transform_sparse(
        sv document
    ) const;

    // ========================================================
    // Accessors
    // ========================================================

    [[nodiscard]]
    sz vocabulary_size() const noexcept;

    [[nodiscard]]
    const Vocabulary& vocabulary() const noexcept;

    [[nodiscard]]
    const vec<double>& idf() const noexcept;

    // ========================================================
    // Model Data & Persistence
    // ========================================================

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

    void collect_term_ids(
        sv document,
        vec<u32>& ids
    ) const;

private:

    Tokenizer tokenizer_;
    Vocabulary vocabulary_;

    sz max_features_{50'000};
    sz min_df_{3};

    sz ngram_min_{1};
    sz ngram_max_{2};

    bool sublinear_tf_{true};

    vec<double> idf_;

    bool fitted_{false};
};

} // namespace sentiment